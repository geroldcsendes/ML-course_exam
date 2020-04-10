library(data.table)
library(caret)
library(MLmetrics)
library(kableExtra)
library(skimr)
library(Boruta) # for finding relevant vars for LM model
library(keras) # for DL


# Define grid for nn hyperparam-tuning
nn_grid <- expand.grid(
  layer = c(3, 5),
  unit_layer = c(30, 50),
  dropout = c(0, 1)
)
nn_grid_ads <- expand.grid(
  layer = c(2),
  unit_layer = c(100),
  dropout = c(0,1)
)

nn_grid <- rbind(nn_grid, nn_grid_ads)
# save grid
#write.csv(nn_grid, 'report/nn_grid.csv', row.names = F)

# load data
# train
x_train <- readRDS('data/nn/x_train_nn.RDS')
y_train <- readRDS('data/nn/y_train_nn.RDS')

# valid
x_valid <- readRDS('data/nn/x_valid_nn.RDS')
y_valid <- readRDS('data/nn/y_valid_nn.RDS')

# test
x_test <- readRDS('data/nn/x_test_nn.RDS')
y_test <- readRDS('data/nn/y_test_nn.RDS')

# For submission data
# load article_id
article_ids <- read.csv('data/test.csv')
article_ids <- article_ids[['article_id']]
# full test set
full_test <- readRDS('data/nn/x_test_full.RDS')

#### Early ####
# set early stopping for all models
es <- callback_early_stopping(monitor = 'val_accuracy', min_delta = 0.0001, patience = 5)


##### Model 1 #####
nn_model1 <- keras_model_sequential() 
nn_model1 %>% 
  layer_dense(units = 50, activation = 'relu', input_shape = 72) %>% 
  layer_dense(units = 50, activation = 'relu') %>% 
  layer_dense(units = 50, activation = 'relu') %>% 
  layer_dense(units = 2, activation = 'sigmoid') # output layer

nn_model1 %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy') # AUC
)
summary(nn_model1)

nn_model1 %>% fit(
  x_train, y_train, 
  epochs = 30, 
  batch_size = 5, 
  validation_data = list(x_valid, y_valid),
  callbacks = es
)

# read model
#nn_model1 <- load_model_tf("models/nn_model1")
nn1_pred_vec <- predict(nn_model1, x_test)

# create df to store results
res_df <- data.frame(
  model = c('nn1'),
  test_AUC = c(AUC(y_pred = nn1_pred_vec[, 2], y_test[, 2]))
)

# predict
model1_pred_vec <- predict(nn_model1, full_test)
model1_preds <- data.frame(
  article_id = article_ids,
  score = model1_pred_vec[, 2]
)

# save model
#nn_model1 %>% save_model_tf("models/nn_model1")
# save pred
write.csv(model1_preds, 'data/nn_preds/nn1.csv', row.names = F)

##### Model 2 #####
nn_model2 <- keras_model_sequential() 
nn_model2 %>% 
  layer_dense(units = 50, activation = 'relu', input_shape = 72) %>% 
  #layer_dropout(rate = 0.25) %>%
  layer_dense(units = 50, activation = 'relu') %>% 
  #layer_dropout(rate = 0.25) %>% 
  layer_dense(units = 50, activation = 'relu') %>% 
  #layer_dropout(rate = 0.25) %>% 
  layer_dense(units = 50, activation = 'relu') %>% 
  layer_dense(units = 50, activation = 'relu') %>% 
  layer_dense(units = 2, activation = 'sigmoid') # output layer

nn_model2 %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy') # AUC
)
summary(nn_model2)

nn_model2 %>% fit(
  x_train, y_train, 
  epochs = 30, 
  batch_size = 5, 
  validation_data = list(x_valid, y_valid),
  callbacks = es
)

# read model
# nn_model2 <- load_model_tf("models/nn_model2")

nn2_pred_vec <- predict(nn_model2, x_test)

# create df to store results
res_df <- rbind(
  res_df, 
  data.frame(
    model = c('nn2'),
    test_AUC = c(AUC(y_pred = nn2_pred_vec[, 2], y_test[, 2]))
  )
)

# predict
model2_pred_vec <- predict(nn_model2, full_test)
model2_preds <- data.frame(
  article_id = article_ids,
  score = model2_pred_vec[, 2]
)

# save model
#nn_model1 %>% save_model_tf("models/nn_model1")
# save pred
write.csv(model2_preds, 'data/nn_preds/nn2.csv', row.names = F)

##### Model 3 #####
nn_model3 <- keras_model_sequential() 
nn_model3 %>% 
  layer_dense(units = 100, activation = 'relu', input_shape = 72) %>% 
  layer_dense(units = 100, activation = 'relu') %>% 
  layer_dense(units = 2, activation = 'sigmoid') # output layer

nn_model3 %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy') # AUC
)
summary(nn_model3)

nn_model3 %>% fit(
  x_train, y_train, 
  epochs = 30, 
  batch_size = 5, 
  validation_data = list(x_valid, y_valid),
  callbacks = es
)

nn3_pred_vec <- predict(nn_model3, x_test)

# create df to store results
res_df <- rbind(
  res_df, 
  data.frame(
    model = c('nn3'),
    test_AUC = c(AUC(y_pred = nn3_pred_vec[, 2], y_test[, 2]))
  )
)

# predict
model3_pred_vec <- predict(nn_model3, full_test)
model3_preds <- data.frame(
  article_id = article_ids,
  score = model3_pred_vec[, 2]
)

# save model
#nn_model1 %>% save_model_tf("models/nn_model1")
# save pred
write.csv(model3_preds, 'data/nn_preds/nn3.csv', row.names = F)

########################
### Apply regularization
#######################

##### REG Model 1 #####
nn_model_reg1 <- keras_model_sequential() 
nn_model_reg1 %>% 
  layer_dense(units = 50, activation = 'relu', input_shape = 72) %>% 
  layer_dropout(rate = 0.15) %>% 
  layer_dense(units = 50, activation = 'relu') %>% 
  layer_dropout(rate = 0.15) %>% 
  layer_dense(units = 50, activation = 'relu') %>% 
  layer_dropout(rate = 0.15) %>% 
  layer_dense(units = 2, activation = 'sigmoid') # output layer

nn_model_reg1 %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy') # AUC
)
summary(nn_model_reg1)

nn_model_reg1 %>% fit(
  x_train, y_train, 
  epochs = 100, 
  batch_size = 5, 
  validation_data = list(x_valid, y_valid),
  callbacks = es
)

nn_reg1_pred_vec <- predict(nn_model_reg1, x_test)

# create df to store results
res_df <- rbind(
  res_df, 
  data.frame(
    model = c('nn_reg1'),
    test_AUC = c(AUC(y_pred = nn_reg1_pred_vec[, 2], y_test[, 2]))
  )
)

# predict
nn_reg1_pred_vec <- predict(nn_model_reg1, full_test)
nn_reg1_preds <- data.frame(
  article_id = article_ids,
  score = nn_reg1_pred_vec[, 2]
)

# save model
#nn_model1 %>% save_model_tf("models/nn_model1")
# save pred
write.csv(nn_reg1_preds, 'data/nn_preds/nn_reg1.csv', row.names = F)

##### REG Model 2 #####

nn_model_reg2 <- keras_model_sequential() 
nn_model_reg2 %>% 
  layer_dense(units = 50, activation = 'relu', input_shape = 72) %>% 
  layer_dropout(rate = 0.15) %>% 
  layer_dense(units = 50, activation = 'relu') %>% 
  layer_dropout(rate = 0.15) %>% 
  layer_dense(units = 50, activation = 'relu') %>% 
  layer_dropout(rate = 0.15) %>% 
  layer_dense(units = 50, activation = 'relu') %>%
  layer_dropout(rate = 0.15) %>% 
  layer_dense(units = 50, activation = 'relu') %>% 
  layer_dropout(rate = 0.15) %>% 
  layer_dense(units = 2, activation = 'sigmoid') # output layer

nn_model_reg2 %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy') # AUC
)
summary(nn_model_reg2)

nn_model_reg2 %>% fit(
  x_train, y_train, 
  epochs = 100, 
  batch_size = 5, 
  validation_data = list(x_valid, y_valid),
  callbacks = es
)

nn_reg2_pred_vec <- predict(nn_model_reg2, x_test)

# create df to store results
res_df <- rbind(
  res_df, 
  data.frame(
    model = c('nn_reg2'),
    test_AUC = c(AUC(y_pred = nn_reg2_pred_vec[, 2], y_test[, 2]))
  )
)

# predict
nn_reg2_pred_vec <- predict(nn_model_reg2, full_test)
nn_reg2_preds <- data.frame(
  article_id = article_ids,
  score = nn_reg2_pred_vec[, 2]
)

# save model
#nn_model1 %>% save_model_tf("models/nn_model1")
# save pred
write.csv(nn_reg2_preds, 'data/nn_preds/nn_reg2.csv', row.names = F)


##### REG Model 3 #####

nn_model_reg3 <- keras_model_sequential() 
nn_model_reg3 %>% 
  layer_dense(units = 100, activation = 'relu', input_shape = 72) %>%
  layer_dropout(rate = 0.15) %>%
  layer_dense(units = 100, activation = 'relu') %>% 
  layer_dropout(rate = 0.15) %>%
  layer_dense(units = 2, activation = 'sigmoid') # output layer

nn_model_reg3 %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy') # AUC
)
summary(nn_model_reg3)

nn_model_reg3 %>% fit(
  x_train, y_train, 
  epochs = 100, 
  batch_size = 5, 
  validation_data = list(x_valid, y_valid),
  callbacks = es
)

nn_reg3_pred_vec <- predict(nn_model_reg3, x_test)

# create df to store results
res_df <- rbind(
  res_df, 
  data.frame(
    model = c('nn_reg3'),
    test_AUC = c(AUC(y_pred = nn_reg3_pred_vec[, 2], y_test[, 2]))
  )
)

# predict
nn_reg3_pred_vec <- predict(nn_model_reg3, full_test)
nn_reg3_preds <- data.frame(
  article_id = article_ids,
  score = nn_reg3_pred_vec[, 2]
)

# save model
#nn_model1 %>% save_model_tf("models/nn_model1")
# save pred
write.csv(nn_reg3_preds, 'data/nn_preds/nn_reg3.csv', row.names = F)

res_df$layers <- c(3, 5, 2, 3, 5, 2)
res_df$units <- c(50, 50, 100, 50, 50, 100)
res_df$reg_method <- c('none', 'none', 'none', '15% droput', '15% droput', '15% droput')
res_df
write.csv(res_df, 'data/nn/nn_performance.csv')