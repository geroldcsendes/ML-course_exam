library(data.table)
library(caret)
library(MLmetrics)
library(kableExtra)
library(skimr)
library(Boruta) # for finding relevant vars for LM model

# load in data
data_train <- readRDS('data/data_train.RDS')
trainTransformed <- readRDS('data/trainTransformed.RDS')

data_test <- readRDS('data/data_test.RDS')
testTransformed <- readRDS('data/testTransformed.RDS')


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

# Train the best tune on whole train - no trainControl!
lasso_model1_grid <- lasso_model$bestTune
set.seed(1)
lasso_model1 <- train(
  is_popular ~.,
  method = "glmnet",
  data=trainTransformed,
  tuneGrid = lasso_model1_grid,
  verbose = FALSE,
  metric = "AUC"
)
lasso_model1_grid

lasso_preds_model1 <- data_test[, 'article_id']
lasso_preds_model1$score  <- predict(lasso_model1, newdata = testTransformed, type="prob")$Yes
write.csv(lasso_preds_model1, 'data/lasso_preds_model1.csv', row.names = F)

## ----lm_pred------------------------------------------------------------------------------------------------
