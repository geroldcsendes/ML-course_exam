library(data.table)
library(caret)
library(MLmetrics)
library(kableExtra)
library(skimr)
library(Boruta) # for finding relevant vars for LM model

# source helper function
source('src/helpers.R')

# load in data
data_train <- readRDS('data/data_train.RDS')
data_test <- readRDS('data/data_test.RDS')

## ----RF-----------------------------------------------------------------------------------------------------
# Cross validaiton
# set CV as train control
train_control <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = multiClassSummary # needed for AUC metric
)
# Set other train control for final models
train_control_finals <- trainControl(
  classProbs = TRUE,
  summaryFunction = multiClassSummary # needed for AUC metric
)

# set grid
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
# saveRDS(rf_model, '../models/rf_model.rds')
# get top performances
rf_results <- rf_model$results
rf_results <- rf_results[order(- rf_results$AUC),]

# 
# rf_preds <- data_test[, 'article_id']
# rf_preds$score  <- predict(rf_model, newdata = data_test, type="prob")$Yes
# write.csv(rf_preds, '../data/rf_preds.csv', row.names = F)

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


# get top rf models
top_rf <- top_performers(model = rf_model, model_params = c('mtry', 'splitrule', 'min.node.size', 'AUC'))

# re-estimate top rf models
estimate_top_rf_models(tune_grid = top_rf, data_test = data_test, data_train = data_train)


#### New parameter tuning ####

# Best fits are when mtry is at lowest value
# try smaller mtry and wider node size
# set grid
rf_grid2 <- expand.grid(
  mtry = c(3,4),
  splitrule = "gini",
  min.node.size = c(7, 8, 9, 10, 11)
)
# based on first run, extend node size for best mtry fit
rf_grid1_ext <- expand.grid(
  mtry = c(5),
  splitrule = "gini",
  min.node.size = c(5, 6, 12, 13)
)
# merge grids
rf_grid2 <- rbind(rf_grid1_ext, rf_grid2)


#### MODEL RUN NO. 2 ####
set.seed(1)
rf_model_run2 <- train(
  is_popular ~.,
  method = "ranger",
  data=data_train,
  trControl = train_control,
  tuneGrid = rf_grid2,
  verbose = FALSE,
  metric = "AUC",
  importance = 'impurity'
)

# get top rf models in Round 2.
top_rf2 <- top_performers(model = rf_model_run2, model_params = c('mtry', 'splitrule', 'min.node.size', 'AUC'))

# re-estimate top rf models
estimate_top_rf_models(tune_grid = top_rf2, data_test = data_test, data_train = data_train)


## List models of Round1 and Round 2 into a df
rf_pooled_top <- rbind(
  top_rf, top_rf2
)
# order by AUC
rf_pooled_top <- rf_pooled_top[order(-rf_pooled_top$AUC), ]

#######################################################
#### RUN ON RESTRICTED VARIABLE SET PER BORUTA ########
#######################################################

# load in data
boruta_train <- readRDS('data/boruta/boruta_train.RDS')
boruta_test <- readRDS('data/boruta/boruta_test.RDS')

# set boruta grid NO. 1
boruta_rf_grid1 <- expand.grid(
  mtry = c(3, 4, 5),
  splitrule = "gini",
  min.node.size = c(6, 7, 8)
)

#### BORUTA MODEL RUN NO. 1 ####
set.seed(1)
boruta_rf_model_1 <- train(
  is_popular ~.,
  method = "ranger",
  data=boruta_train,
  trControl = train_control,
  tuneGrid = boruta_rf_grid1,
  verbose = T,
  metric = "AUC",
  importance = 'impurity'
)

# Summary: check mtry = 2


#### BORUTA NO. 2.

# set boruta grid NO. 2
boruta_rf_grid2 <- expand.grid(
  mtry = c(2),
  splitrule = "gini",
  min.node.size = c(5, 6, 7, 8)
)

#### BORUTA MODEL RUN NO. 2 ####
set.seed(1)
boruta_rf_model_2 <- train(
  is_popular ~.,
  method = "ranger",
  data=boruta_train,
  trControl = train_control,
  tuneGrid = boruta_rf_grid2,
  verbose = T,
  metric = "AUC",
  importance = 'impurity'
)

# SUMMARY: mtry=2 is a lot better fit than bigger values

# get top rf models in Round 2.
top_boruta2 <- top_performers(model = boruta_rf_model_2, 
                              model_params = c('mtry', 'splitrule', 'min.node.size', 'AUC'))
top_boruta2 <- top_boruta2[c(1, 2), ]
# re-estimate top rf models
estimate_top_rf_models(tune_grid = top_boruta2, data_test = data_test, data_train = data_train)



