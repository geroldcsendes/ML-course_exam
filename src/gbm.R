library(data.table)
library(caret)
library(MLmetrics)
library(kableExtra)
library(skimr)
library(Boruta) # for finding relevant vars for LM model

# load in data
data_train <- readRDS('data/data_train.RDS')
data_test <- readRDS('data/data_test.RDS')

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
