##############################
# script for tuning xgbm for kaggle challenge
##############################

library(data.table)
library(caret)
library(MLmetrics)
library(kableExtra)
library(skimr)
library(Boruta) # for finding relevant vars for LM model

# source helper function
source('src/helpers.R')

# xgbTree modelinfo
modelLookup('xgbTree')

# load in data
data_train <- readRDS('data/data_train.RDS')
data_test <- readRDS('data/data_test.RDS')

## ----XGBM-----------------------------------------------------------------------------------------------------
# Cross validaiton
# set CV as train control - 3Fold
train_control <- trainControl(
  method = "cv",
  number = 3,
  classProbs = TRUE,
  summaryFunction = multiClassSummary # needed for AUC metric
)
# Set other train control for final models
train_control_finals <- trainControl(
  classProbs = TRUE,
  summaryFunction = multiClassSummary # needed for AUC metric
)

################
# set grid No. 1
xgbm_grid <- expand.grid(
  nrounds = c(1500, 2000, 3000),
  max_depth = c(5,6,9),
  eta = c(0.01), #0.1
  gamma = c(0.0002),
  colsample_bytree = c(0.7),
  min_child_weight = c(1),
  subsample = c(0.7)
)

set.seed(1)
xgbm_model1 <- train(
  is_popular ~.,
  method = "xgbTree",
  data=data_train,
  trControl = train_control,
  tuneGrid = xgbm_grid,
  verbose = T,
  metric = "AUC"
)
# save model
saveRDS(xgbm_model1, 'models/xgbm1.RDS')

# looks like bigger trees are not worth it
# also deep trees not really worth it

################
# set grid No. 2
xgbm_grid2 <- expand.grid(
  nrounds = c(1500, 2000),
  max_depth = c(5, 6, 7),
  eta = c(0.01), #0.1
  gamma = c(0.0002),
  colsample_bytree = c(0.5, 0.7, 1),
  min_child_weight = c(0.5, 1, 1.5),
  subsample = c(0.7)
)

# Train Round 2
set.seed(1)
xgbm_model2 <- train(
  is_popular ~.,
  method = "xgbTree",
  data=data_train,
  trControl = train_control,
  tuneGrid = xgbm_grid2,
  verbose = T,
  metric = "AUC"
)
# save model
saveRDS(xgbm_model2, 'models/xgbm2.RDS')


# get top gbm models for FIRST run
top_xgbm1 <- top_performers(model = xgbm_model1, 
            model_params = c('nrounds', 'max_depth', 'eta', 'gamma', 'colsample_bytree',
                             'min_child_weight', 'subsample', 'AUC'), criterion = 0.003)
top_xgbm1$model <- c("xgbm_model1")
  
# get top gbm models for SECOND run
top_xgbm2 <- top_performers(model = xgbm_model2, 
                            model_params = c('nrounds', 'max_depth', 'eta', 'gamma', 'colsample_bytree',
                                             'min_child_weight', 'subsample', 'AUC'), criterion = 0.003)
top_xgbm2$model <- c("xgbm_model2")

# SUMMARY: top_xgbm1 is subsample of top1
# max depth =5 seems really convincing
# try max_depth = 4
# more trees don't really help

################
# set grid No. 3
xgbm_grid3 <- expand.grid(
  nrounds = c(1500),
  max_depth = c(4),
  eta = c(0.01), #0.1
  gamma = c(0.0002),
  colsample_bytree = c(0.5, 0.7, 1),
  min_child_weight = c(0.5, 1, 1.5),
  subsample = c(0.7)
)

# Train Round 3
set.seed(1)
xgbm_model3 <- train(
  is_popular ~.,
  method = "xgbTree",
  data=data_train,
  trControl = train_control,
  tuneGrid = xgbm_grid3,
  verbose = T,
  metric = "AUC"
)
# save model
saveRDS(xgbm_model3, 'models/xgbm3.RDS')

top_xgbm3 <- top_performers(model = xgbm_model3, 
                            model_params = c('nrounds', 'max_depth', 'eta', 'gamma', 'colsample_bytree',
                                             'min_child_weight', 'subsample', 'AUC'), criterion = 0.003)
top_xgbm3$model <- c("xgbm_model3")

# SUMMARY: it may still be worth experimenting w. min_child_weight

################
# set grid No. 4
xgbm_grid4 <- expand.grid(
  nrounds = c(1500),
  max_depth = c(4),
  eta = c(0.01), #0.1
  gamma = c(0.0002),
  colsample_bytree = c(0.5, 0.7, 1),
  min_child_weight = c(0.3, 1.75),
  subsample = c(0.7)
)

# Train Round 4
set.seed(1)
xgbm_model4 <- train(
  is_popular ~.,
  method = "xgbTree",
  data=data_train,
  trControl = train_control,
  tuneGrid = xgbm_grid4,
  verbose = T,
  metric = "AUC"
)
# save model
saveRDS(xgbm_model4, 'models/xgbm4.RDS')
top_xgbm4 <- top_performers(model = xgbm_model4, 
                            model_params = c('nrounds', 'max_depth', 'eta', 'gamma', 'colsample_bytree',
                                             'min_child_weight', 'subsample', 'AUC'), criterion = 0.003)
top_xgbm4$model <- c("xgbm_model4")

# SUMMARY: seems like not so "extreme" min_child_weight are better
# now let's check depth of 3

################
# set grid No. 5
xgbm_grid5 <- expand.grid(
  nrounds = c(1500),
  max_depth = c(3),
  eta = c(0.01), #0.1
  gamma = c(0.0002),
  colsample_bytree = c(0.5, 0.7, 1),
  min_child_weight = c(0.5, 1, 1.5), 
  subsample = c(0.7)
)

# Train Round 5
set.seed(1)
xgbm_model5 <- train(
  is_popular ~.,
  method = "xgbTree",
  data=data_train,
  trControl = train_control,
  tuneGrid = xgbm_grid5,
  verbose = T,
  metric = "AUC"
)
# save model
saveRDS(xgbm_model5, 'models/xgbm5.RDS')
top_xgbm5 <- top_performers(model = xgbm_model5, 
                            model_params = c('nrounds', 'max_depth', 'eta', 'gamma', 'colsample_bytree',
                                             'min_child_weight', 'subsample', 'AUC'), criterion = 0.003)
top_xgbm5$model <- c("xgbm_model5")



########## XGBM FINALISTS #########
xgbm_pooled_top <- rbind(
  top_xgbm1, top_xgbm2, top_xgbm3, top_xgbm4, top_xgbm5
)
# order by AUC
xgbm_pooled_top <- xgbm_pooled_top[order(-xgbm_pooled_top$AUC), ]


#### Make predictions
#### top2: WITH REFITTING
# re-estimate top rf models: TOP3
estimate_top_xgbm_models(tune_grid = xgbm_pooled_top[c(3, 4), ], 
                         data_test = data_test, data_train = data_train)

#######################################################
#### RUN ON RESTRICTED VARIABLE SET PER BORUTA ########
#######################################################

# load in data
boruta_train <- readRDS('data/boruta/boruta_train.RDS')
boruta_test <- readRDS('data/boruta/boruta_test.RDS')

# set boruta grid NO. 1
boruta_xgbm_grid1 <- expand.grid(
  nrounds = c(1500),
  max_depth = c(3, 4),
  eta = c(0.01), #0.1
  gamma = c(0.0002),
  colsample_bytree = c(0.3, 0.5, 0.7),
  min_child_weight = c(0.5, 1, 1.5), 
  subsample = c(0.7)
)

#### BORUTA MODEL RUN NO. 1 ####
boruta_xgbm_model_1 <- train(
  is_popular ~.,
  method = "xgbTree",
  data=boruta_train,
  trControl = train_control,
  tuneGrid = boruta_xgbm_grid1,
  verbose = T,
  metric = "AUC"
)

top_boruta1 <- top_performers(model = boruta_xgbm_model_1, 
                              model_params = c('nrounds', 'max_depth', 'eta', 'gamma', 'colsample_bytree',
                                               'min_child_weight', 'subsample', 'AUC'), criterion = 0.003)
# write for report
write.csv(top_boruta1, 'data/xgbm_performance/boruta1.csv') 

# get top models in boruta

###################
##### Boruta No. 2
###################

# let's try max_depth = 5
# set boruta grid NO. 1
boruta_xgbm_grid2 <- expand.grid(
  nrounds = c(1500),
  max_depth = c(5),
  eta = c(0.01), #0.1
  gamma = c(0.0002),
  colsample_bytree = c(0.3, 0.5, 0.7),
  min_child_weight = c(0.5, 1, 1.5), 
  subsample = c(0.7)
)

#### BORUTA MODEL RUN NO. 2 ####
boruta_xgbm_model_2 <- train(
  is_popular ~.,
  method = "xgbTree",
  data=boruta_train,
  trControl = train_control,
  tuneGrid = boruta_xgbm_grid2,
  verbose = T,
  metric = "AUC"
)

top_boruta2 <- top_performers(model = boruta_xgbm_model_2, 
                              model_params = c('nrounds', 'max_depth', 'eta', 'gamma', 'colsample_bytree',
                                               'min_child_weight', 'subsample', 'AUC'), criterion = 0.003)
top_boruta2 
# write for report
write.csv(top_boruta2, 'data/xgbm_performance/boruta2.csv') 

# get top rf models in bor
top_rf2 <- top_performers(model = rf_model_run2, model_params = c('mtry', 'splitrule', 'min.node.size', 'AUC'))



###################
##### Boruta No. 3
###################

# max_depth = 6
boruta_xgbm_grid3 <- expand.grid(
  nrounds = c(1500),
  max_depth = c(6),
  eta = c(0.01), #0.1
  gamma = c(0.0002),
  colsample_bytree = c(0.3, 0.5, 0.7),
  min_child_weight = c(0.5, 1, 1.5), 
  subsample = c(0.7)
)

#### BORUTA MODEL RUN NO. 2 ####
boruta_xgbm_model_3 <- train(
  is_popular ~.,
  method = "xgbTree",
  data=boruta_train,
  trControl = train_control,
  tuneGrid = boruta_xgbm_grid3,
  verbose = T,
  metric = "AUC"
)

top_boruta3 <- top_performers(model = boruta_xgbm_model_3, 
                              model_params = c('nrounds', 'max_depth', 'eta', 'gamma', 'colsample_bytree',
                                               'min_child_weight', 'subsample', 'AUC'), criterion = 0.003)
top_boruta3

# write boruta results
write.csv(top_boruta3, 'data/xgbm_performance/boruta3.csv')

########## XGBM BOURUTA FINALISTS #########
xgbm_boruta_pooled_top <- rbind(
  top_boruta1, top_boruta2, top_boruta3
)
# order by AUC
xgbm_boruta_pooled_top <- xgbm_boruta_pooled_top[order(-xgbm_boruta_pooled_top$AUC), ]
xgbm_boruta_pooled_top
write.csv(xgbm_boruta_pooled_top, 'data/xgbm_performance/top_boruta.csv')

#### Make predictions
#### top2: WITH REFITTING
# re-estimate top boruta models: TOP3
estimate_top_xgbm_models(tune_grid = xgbm_boruta_pooled_top[c(1, 2, 3), ], 
                         data_test = data_test, data_train = data_train)

#modelLookup(model='mlpKerasDropout')



###################
##### Boruta No. 4
###################

# max_depth = 6
boruta_xgbm_grid4 <- expand.grid(
  nrounds = c(2000),
  max_depth = c(4, 5),
  eta = c(0.01), #0.1
  gamma = c(0.0002),
  colsample_bytree = c(0.2, 0.4, 0.6, 0.8, 0.9),
  min_child_weight = c(0.3, 0.4, 0.6, 0.9, 1.2, 1.5), 
  subsample = c(0.7)
)

#### BORUTA MODEL RUN NO. 2 ####
boruta_xgbm_model_4 <- train(
  is_popular ~.,
  method = "xgbTree",
  data=boruta_train,
  trControl = train_control,
  tuneGrid = boruta_xgbm_grid4,
  verbose = T,
  metric = "AUC"
)
top_boruta4 <- top_performers(model = boruta_xgbm_model_4, 
                              model_params = c('nrounds', 'max_depth', 'eta', 'gamma', 'colsample_bytree',
                                               'min_child_weight', 'subsample', 'AUC'), criterion = 0.0005)
dim(top_boruta4)
estimate_top_xgbm_models(tune_grid = top_boruta4[c(1, 2, 3), ], 
                         data_test = data_test, data_train = data_train)

