library(data.table)
library(GGally)
library(h2o)

#h2o.init()
# stop it via h2o.shutdown()
h2o.init(max_mem_size = '4g')


###### Load data for different models
# data for gbm
boruta_train <- readRDS('data/boruta/boruta_train.RDS')
boruta_test <- readRDS('data/boruta/boruta_test.RDS')

# data for glm and nnet
trainTransformed <- readRDS('data/trainTransformed.RDS')
testTransformed <- readRDS('data/testTransformed.RDS')

# convert to h2o dataű
# GBM
gbm_data <- as.h2o(boruta_train)
gbm_data_split <- h2o.splitFrame(gbm_data, ratios = 0.15, seed = 123)
gbm_train <- gbm_data_split[[1]]
gbm_holdout <- gbm_data_split[[2]]

#  Linear

# DL
dl_data <- as.h2o(trainTransformed)
dl_data_split <- h2o.splitFrame(dl_data, ratios = 0.15, seed = 123)
dl_train <- dl_data_split[[1]]
dl_holdout <- dl_data_split[[2]]


##### Train models

# GBM
y_gbm <- "is_popular"
X_gbm <- setdiff(names(boruta_train), y_gbm)

h2o.table(gbm_train['is_popular'])


gbm_params <- list(learn_rate = c(0.01),
                   max_depth = c(3, 4, 5, 6),
                   sample_rate = c(0.85),
                   col_sample_rate = c(0.2, 0.4, 0.6, 0.8, 0.9)
                   )


gbm_grid <- h2o.grid(x =  X_gbm, 
                     y = y_gbm,
                     training_frame = gbm_train,
                     validation_frame = gbm_holdout,
                     algorithm = "gbm", 
                     nfolds = 3,
                     seed = 123,
                     ntrees = 2000,
                     hyper_params = gbm_params,
                     sample_rate_per_class = c(0.8 ,1), # downsample class one = 'No'
                     stopping_tolerance = c(0.0002),
                     stopping_metric = c('AUC'),
                     stopping_rounds = c(20),
                     keep_cross_validation_predictions = TRUE)


h2o_run1
h2o_run2 <- h2o.getGrid(gbm_grid@grid_id, sort_by = "AUC", decreasing = T )

gbm_model <- h2o.getModel(h2o.getGrid(gbm_grid@grid_id, sort_by = "AUC", decreasing = T )@model_ids[[1]])
h2o.auc(gbm_model, train=TRUE, valid=TRUE, xval=T)


print(h2o.performance(gbm_model, xval = TRUE))
h2o.auc(h2o.performance(gbm_model, newdata = gbm_holdout))


# DL
y_dl <- "is_popular"
X_dl <- setdiff(names(trainTransformed), y_gbm)

h2o.table(gbm_train['is_popular'])

dl_params= list(
    hidden=list(c(150, 150, 150), c(100, 100, 100)),
    rate = 0.01,
    hidden_dropout_ratios = list(c(0.2, 0.2, 0.2), c(0.3, 0.3, 0.3), c(0.5, 0.5, 0.5), c(0.6, 0.6, 0.6)),
    activation = c('RectifierWithDropout'),
    epochs = c(100),
    stopping_rounds = c(6),
    stopping_metric = c('AUC'),
    stopping_tolerance = c(0.0008)
    )

    
deeplearning_model <- h2o.grid(
        x = X_dl, y= y_dl,
        algorithm = 'deeplearning',
        training_frame = dl_train,
        validation_frame = dl_holdout,
        hyper_params = dl_params,
        seed = 123,
        mini_batch_size = 5,
        keep_cross_validation_predictions = TRUE
)

h2o.getGrid(deeplearning_model@grid_id, sort_by = "AUC", decreasing = T )


# Linae model
glm_model <- h2o.glm(
  x = X_dl, y= y_dl,
  training_frame = dl_train,
  validation_frame = dl_holdout,
  family = "binomial",
  alpha = 1, 
  lambda_search = TRUE,
  seed = 123,
  nfolds = 5, 
  keep_cross_validation_predictions = TRUE  # this is necessary to perform later stacking
)

h2o.getModel(glm_model)

ensemble_model <- h2o.stackedEnsemble(
  X, y,
  training_frame = data_train,
  base_models = list(glm_model, 
                     gbm_model,
                     deeplearning_model),
  keep_levelone_frame = TRUE)

