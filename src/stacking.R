library(data.table)
library(caret)
library(MLmetrics)
library(kableExtra)
library(skimr)
library(Boruta) # for finding relevant vars for LM model
library(keras) # for DL
library(GGally)
library(tidyverse)


## READ IN predictions
# read nn preds
nn1 <- read.csv('data/nn_preds/nn1.csv')
colnames(nn1) <- c('article_id', 'nn1')

nn2 <- read.csv('data/nn_preds/nn2.csv')
colnames(nn2) <- c('article_id', 'nn2')

nn3 <- read.csv('data/nn_preds/nn3.csv')
colnames(nn3) <- c('article_id', 'nn3')

nn_reg1 <- read.csv('data/nn_preds/nn_reg1.csv')
colnames(nn_reg1) <- c('article_id', 'nn_reg1')

nn_reg2 <- read.csv('data/nn_preds/nn_reg2.csv')
colnames(nn_reg2) <- c('article_id', 'nn_reg2')

nn_reg3 <- read.csv('data/nn_preds/nn_reg3.csv')
colnames(nn_reg3) <- c('article_id', 'nn_reg3')

# xgbm
xgbm1 <- read.csv('data/xgbm_preds/nrounds1500_max_depth4_eta0.01_gamma2e-04_colsample_bytree1_min_child_weight0.5_subsample0.7.csv')
colnames(xgbm1) <- c('article_id', 'xgbm1')

xgbm2 <- read.csv('data/xgbm_preds/nrounds1500_max_depth4_eta0.01_gamma2e-04_colsample_bytree0.7_min_child_weight1.75_subsample0.7.csv')
colnames(xgbm2) <- c('article_id', 'xgbm2')

xgbm3 <- read.csv('data/xgbm_preds/nrounds1500_max_depth4_eta0.01_gamma2e-04_colsample_bytree0.7_min_child_weight0.3_subsample0.7.csv')
colnames(xgbm3) <- c('article_id', 'xgbm3')

xgbm4 <- read.csv('data/xgbm_preds/nrounds1500_max_depth4_eta0.01_gamma2e-04_colsample_bytree0.5_min_child_weight1.5_subsample0.7.csv')
colnames(xgbm4) <- c('article_id', 'xgbm4')

# xgbm boruta
xgbm_boruta1 <- read.csv('data/xgbm_preds_boruta/nrounds1500_max_depth5_eta0.01_gamma2e-04_colsample_bytree0.3_min_child_weight0.5_subsample0.7.csv')
colnames(xgbm_boruta1) <- c('article_id', 'xgbm_boruta1')

xgbm_boruta2 <- read.csv('data/xgbm_preds_boruta/nrounds1500_max_depth5_eta0.01_gamma2e-04_colsample_bytree0.5_min_child_weight1_subsample0.7.csv')
colnames(xgbm_boruta2) <- c('article_id', 'xgbm_boruta2')

xgbm_boruta3 <- read.csv('data/xgbm_preds_boruta/nrounds1500_max_depth5_eta0.01_gamma2e-04_colsample_bytree0.3_min_child_weight1.5_subsample0.7.csv')
colnames(xgbm_boruta3) <- c('article_id', 'xgbm_boruta3')

# rf
rf1 <- read.csv('data/rf_preds/mtry5_splitrule1_min.node.size8.csv')
colnames(rf1) <-  c('article_id', 'rf1')

rf2 <- read.csv('data/rf_preds/mtry5_splitrule1_min.node.size10.csv')
colnames(rf2) <-  c('article_id', 'rf2')

rf3 <- read.csv('data/rf_preds/mtry5_splitrule1_min.node.size9.csv')
colnames(rf3) <-  c('article_id', 'rf3')

# rf boruta
rf_boruta1 <- read.csv('data/rf_preds/mtry2_splitrule1_min.node.size6.csv')
colnames(rf_boruta1) <-  c('article_id', 'rf_boruta1')

rf_boruta2 <- read.csv('data/rf_preds/mtry2_splitrule1_min.node.size8.csv')
colnames(rf_boruta2) <-  c('article_id', 'rf_boruta2')

# linear
linear1 <- read.csv('data/lasso_preds_model1.csv')
colnames(linear1) <- c('article_id', 'linear1')
### merge into one df
stacked_test <- list(
  rf_boruta1,
  rf_boruta2,
  rf1,
  rf2,
  rf3,
  xgbm_boruta1,
  xgbm_boruta2,
  xgbm_boruta3,
  xgbm1,
  xgbm2,
  xgbm3,
  xgbm4,
  nn1,
  nn2,
  nn3,
  nn_reg1,
  nn_reg2,
  nn_reg3,
  linear1) %>% reduce(inner_join, by = 'article_id')
View(stacked_df)

# Get training data
train_control_finals <- trainControl(
  method = "none",
  classProbs = TRUE,
  summaryFunction = multiClassSummary # needed for AUC metric
)

# read in data
boruta_train <- readRDS('data/boruta/boruta_train.RDS')
boruta_test <- readRDS('data/boruta/boruta_test.RDS')

data_train <- readRDS('data/data_train.RDS')
data_test <- readRDS('data/data_test.RDS')

# RF boruta
rf_boruta1_model <- readRDS('models/rf/mtry2_splitrule1_min.node.size6.RDS')
rf_boruta1_preds <- predict(rf_boruta1_model, newdata = data_train, type='prob')$Yes

# RF normal
rf1_model <- readRDS('models/rf/mtry5_splitrule1_min.node.size8.RDS')
rf1_preds <- predict(rf1_model, newdata = data_train, type ='prob')$Yes

# XGBM boruta
xgbm_boruta1_model <- readRDS('models/xgbm_boruta/nrounds1500_max_depth5_eta0.01_gamma2e-04_colsample_bytree0.3_min_child_weight0.5_subsample0.7.RDS')
xgbm_boruta1_preds <-  predict(xgbm_boruta1_model, newdata = data_train, type='prob')$Yes

# XGBM
xgbm1_model <- readRDS('models/xgbm/nrounds1500_max_depth4_eta0.01_gamma2e-04_colsample_bytree1_min_child_weight0.5_subsample0.7.RDS')
xgbm1_preds <- predict(xgbm1_model, newdata = data_train, type='prob')$Yes

# NN reg
nn_reg1_preds <- read.csv('data/nn/nn_reg1_train_pred.csv')

# Lasso
linear1_model <- readRDS('models/lasso_model.RDS')
linea1_preds <- predict(linear1_model, newdata = data_train, type='prob')$Yes

# train data
stacked_train <- data.frame(
  rf_boruta1 = rf_boruta1_preds,
  rf1 = rf1_preds,
  xgbm_boruta1 = xgbm_boruta1_preds,
  xgbm1 = xgbm1_preds,
  nn_reg1 = nn_reg1_preds$score,
  linear1 = linea1_preds,
  is_popular = data_train$is_popular
)

## Corr matrix
ggcorr(stacked_df,label = T)
ggsave('report/corr_matrix.png')


# Defne variable sets
X1 <- c('rf_boruta1', 'xgbm_boruta1', 'nn_reg1', 'linear1')
X2 <- c('xgbm_boruta1', 'nn_reg1', 'linear1')
X3 <- c('xgbm_boruta1', 'nn_reg1')
X4 <- c('xgbm_boruta1', 'linear1')
X5 <- c('rf_boruta1', 'nn_reg1', 'linear1')
X6 <- c('rf_boruta1',  'linear1')
X7 <- c('rf_boruta1', 'nn_reg1',)


### Build models
## first build 'on purpose' then by randomizing
# Cross validaiton
# set CV as train control
train_control <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = multiClassSummary # needed for AUC metric
)


##### Stack No. 1
set.seed(1)
stacked1 <- train(
  formula(paste("is_popular ~", paste(X1, collapse = ' + '))),
  method = "glmnet",
  data = stacked_train,
  #trControl = train_control,
  verbose = FALSE,
  metric = "AUC"
)

stacked1_pred_df <-data.frame(
  article_id = data_test$article_id,
  score = predict(stacked1, newdata = stacked_test, type = 'prob')$Yes
)
write.csv(stacked1_pred_df, 'data/stacking/stacked1.csv', row.names = F)


##### Stack No. 2
set.seed(1)
stacked2 <- train(
  formula(paste("is_popular ~", paste(X2, collapse = ' + '))),
  method = "glmnet",
  data = stacked_train,
  #trControl = train_control,
  verbose = FALSE,
  metric = "AUC"
)

stacked2_pred_df <-data.frame(
  article_id = data_test$article_id,
  score = predict(stacked2, newdata = stacked_test, type = 'prob')$Yes
)
write.csv(stacked2_pred_df, 'data/stacking/stacked2.csv', row.names = F)

##### Stack No. 3
set.seed(1)
stacked3 <- train(
  formula(paste("is_popular ~", paste(X3, collapse = ' + '))),
  method = "glmnet",
  data = stacked_train,
  #trControl = train_control,
  verbose = FALSE,
  metric = "AUC"
)

stacked3_pred_df <-data.frame(
  article_id = data_test$article_id,
  score = predict(stacked3, newdata = stacked_test, type = 'prob')$Yes
)
write.csv(stacked3_pred_df, 'data/stacking/stacked3.csv', row.names = F)


##### Stack No. 4
set.seed(1)
stacked4 <- train(
  formula(paste("is_popular ~", paste(X4, collapse = ' + '))),
  method = "glmnet",
  data = stacked_train,
  #trControl = train_control,
  verbose = FALSE,
  metric = "AUC"
)

stacked4_pred_df <-data.frame(
  article_id = data_test$article_id,
  score = predict(stacked4, newdata = stacked_test, type = 'prob')$Yes
)
write.csv(stacked4_pred_df, 'data/stacking/stacked4.csv', row.names = F)

##### Stack No. 5
set.seed(1)
stacked5 <- train(
  formula(paste("is_popular ~", paste(X5, collapse = ' + '))),
  method = "glmnet",
  data = stacked_train,
  #trControl = train_control,
  verbose = FALSE,
  metric = "AUC"
)

stacked5_pred_df <-data.frame(
  article_id = data_test$article_id,
  score = predict(stacked5, newdata = stacked_test, type = 'prob')$Yes
)
write.csv(stacked5_pred_df, 'data/stacking/stacked5.csv', row.names = F)

##### Stack No. 6
set.seed(1)
stacked6 <- train(
  formula(paste("is_popular ~", paste(X6, collapse = ' + '))),
  method = "glmnet",
  data = stacked_train,
  #trControl = train_control,
  verbose = FALSE,
  metric = "AUC"
)

stacked6_pred_df <-data.frame(
  article_id = data_test$article_id,
  score = predict(stacked6, newdata = stacked_test, type = 'prob')$Yes
)
write.csv(stacked6_pred_df, 'data/stacking/stacked6.csv', row.names = F)

##### Stack No. 7
set.seed(1)
stacked7 <- train(
  formula(paste("is_popular ~", paste(X6, collapse = ' + '))),
  method = "glmnet",
  data = stacked_train,
  #trControl = train_control,
  verbose = FALSE,
  metric = "AUC"
)

stacked7_pred_df <-data.frame(
  article_id = data_test$article_id,
  score = predict(stacked7, newdata = stacked_test, type = 'prob')$Yes
)
write.csv(stacked7_pred_df, 'data/stacking/stacked7.csv', row.names = F)
