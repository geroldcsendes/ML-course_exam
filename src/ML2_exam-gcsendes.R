## ----setup, include=FALSE-----------------------------------------------------------------------------------
knitr::opts_chunk$set(echo = TRUE)


## ----libs, echo=FALSE, message=FALSE, warning=FALSE---------------------------------------------------------
library(data.table)
library(caret)
library(MLmetrics)
library(kableExtra)
library(skimr)
library(Boruta) # for finding relevant vars for LM model


## ----data_read----------------------------------------------------------------------------------------------
# Read in data
data_train <- fread('../data/train.csv')
data_test <- fread('../data/test.csv')

# take a basic look at data
dim(data_train)
dim(data_test)

skim(data_train)

# get rid of non-predictive vars
data_train[, c('url', 'timedelta', 'article_id'):= NULL]
# for test article_is needed
data_test[, c('url', 'timedelta'):= NULL]

# select factor vars
factor_vars <- colnames(data_train)
factor_vars <- factor_vars[grepl('is', factor_vars)]

# convert to factor
data_train[, (factor_vars) := lapply(.SD, factor), .SDcols = factor_vars]
# for binary prediction, the target variable must be a factor with levels
levels(data_train$is_popular) <- c("No", "Yes")


# convert test to factor as well
# is_popular is not in test
factor_vars_test <- setdiff(factor_vars, 'is_popular')
data_test[, (factor_vars_test) := lapply(.SD, factor), .SDcols = factor_vars_test]
# str(data_test)

# Cross validaiton
# set CV as train control
train_control <- trainControl(
  method = "cv",
  number = 8,
  classProbs = TRUE,
  summaryFunction = multiClassSummary # needed for AUC metric
)

# Set other train control for final models
train_control_finals <- trainControl(
  classProbs = TRUE,
  summaryFunction = multiClassSummary # needed for AUC metric
)


## ----scale--------------------------------------------------------------------------------------------------
# train data
preProcValues  <- preProcess(data_train, method=c('center','scale'))
trainTransformed <- predict(preProcValues, data_train)

# test data
# IMPORTANT: transform test via train transformation values
testTransformed <- predict(preProcValues, data_test)


## ----lm_prep------------------------------------------------------------------------------------------------
# Only use a subset of data, otherwise runs forever..
set.seed(1)
boruta_data<- data_train[sample(nrow(data_train), 2000), ]
set.seed(1)
boruta_train <- Boruta(
            is_popular ~.,
            data = boruta_data,
            doTrace = 3
)

# Take a look at the result
boruta_train
boruta_decision <- boruta_train$finalDecision
names(boruta_decision[boruta_decision == 'Tentative'])
getSelectedAttributes(boruta_train, withTentative = F)

boruta_df <- attStats(boruta_train)
boruta_df[order(-boruta_df$medianImp),]
boruta_df_factor <- boruta_df[factor_vars,]
boruta_df_factor[order(-boruta_df_factor$medianImp),]



## ----lm-----------------------------------------------------------------------------------------------------
factor_ints <- c('data_channel_is_world', 'data_channel_is_tech', 'data_channel_is_lifestyle')
num_ints <-  c('kw_avg_avg', 'kw_max_avg', 'kw_min_avg')
lasso_interactions <- c()
for (factor_var in factor_ints) {
        for (num_var in num_ints) {
            lasso_interactions <- c(lasso_interactions, paste0(factor_var, ' * ', num_var))
        }
}

# define xvars for LASSO
X_lasso <- setdiff(colnames(data_train), 'is_popular')
X_lasso <- c(X_lasso, lasso_interactions)

# define tunegrid
lambdas <- 10^seq(-1, -5, by = -1)
lasso_tune_grid <- expand.grid(
  "alpha" = c(1),
  "lambda" = c(lambdas, lambdas / 2) 
)

# create model
lasso_model <- train(
    formula(paste0("is_popular ~ ", paste0(X_lasso, collapse = " + "))),
    data = trainTransformed,
    method = "glmnet",
    trControl = train_control,
    tuneGrid = lasso_tune_grid,
    metric = 'AUC'
)

lasso_model
saveRDS(lasso_model, '../models/lasso_model.rds')


## ----lm_pred------------------------------------------------------------------------------------------------
lasso_preds <- data_test[, 'article_id']
lasso_preds$score  <- predict(lasso_model, newdata = testTransformed, type="prob")$Yes
write.csv(lasso_preds, '../data/lm_preds.csv', row.names = F)


## ----RF-----------------------------------------------------------------------------------------------------
rf_grid <- expand.grid(
  mtry = c(5, 7, 8, 9),
  splitrule = "gini",
  min.node.size = c(7, 8, 9, 10, 11)
)

set.seed(1)
rf_model <- train(
      is_popular ~.,
      method = "ranger",
      data=data_train,
      trControl = train_control,
      tuneGrid = rf_grid,
      verbose = FALSE,
      metric = "AUC",
      importance = 'impurity'
)

# save model
saveRDS(rf_model, '../models/rf_model.rds')
# get top performances
rf_results <- rf_model$results
rf_results <- rf_results[order(- rf_results$AUC),]


rf_preds <- data_test[, 'article_id']
rf_preds$score  <- predict(rf_model, newdata = data_test, type="prob")$Yes
write.csv(rf_preds, '../data/rf_preds.csv', row.names = F)

# number one model
rf_model1_grid <- rf_model$bestTune
set.seed(1)
rf_model1 <- train(
      is_popular ~.,
      method = "ranger",
      data=data_train,
      tuneGrid = rf_model1_grid,
      trControl = train_control_finals,
      verbose = FALSE,
      metric = "AUC",
      importance = 'impurity'
)
rf_model1_grid

rf_preds_model1 <- data_test[, 'article_id']
rf_preds_model1$score  <- predict(rf_model1, newdata = data_test, type="prob")$Yes
write.csv(rf_preds_model1, '../data/rf_preds_model1.csv', row.names = F)


## ----GBM----------------------------------------------------------------------------------------------------
# Train GBM
gbm_grid <- expand.grid(n.trees = c(300, 500), 
                        interaction.depth = c(2, 3, 5), 
                        shrinkage = c(0.005, 0.01, 0.1),
                        n.minobsinnode = c(5, 7, 9, 11))
set.seed(1)
gbm_model <- train(is_popular ~.,
                   method = "gbm",
                   data = data_train,
                   trControl = train_control,
                   tuneGrid = gbm_grid,
                   verbose = FALSE, 
                   metric = 'AUC'
                   )
saveRDS(gbm_model, '../models/gbm_model.rds')

gbm_preds <- data_test[, 'article_id']
gbm_preds$score  <- predict(gbm_model, newdata = data_test, type="prob")$Yes
write.csv(gbm_preds, '../data/gbm_preds.csv', row.names = F)


## ----NN-----------------------------------------------------------------------------------------------------



## -----------------------------------------------------------------------------------------------------------


