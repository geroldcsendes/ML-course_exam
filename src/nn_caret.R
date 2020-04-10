library(data.table)
library(caret)
library(MLmetrics)
library(kableExtra)
library(skimr)
library(Boruta) # for finding relevant vars for LM model
library(keras) # for DL
library(recipes)


getModelInfo('mlpKerasDropout')
# retrain nn model(s) in caret for later stacking in caretStack()

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


# convert matrices to dataframe
# train
x_caret_train <- as.data.frame(x_train)
y_caret_train <- as.data.frame(y_train)

# valid
x_caret_valid <- as.data.frame(x_valid)
y_caret_valid <- as.data.frame(y_valid)

# test
x_caret_test <- as.data.frame(x_test)
y_caret_test <- as.data.frame(y_test)

# getModelInfo('mlpKerasDropout')
######### TRAIN Model#############
nn_caret <- train(
            x= x_train, y = y_train, 
            method = "mlpKerasDropout",
            metric = 'logLoss',
            loss = 'binary_crossentropy',
            optimizer = 'adam',
            size = c(50, 50, 50),
            dropout = c(0.1, 0.1, .1),
            batch_size = ,
            activation = c('relu', 'relu', 'relu', 'sigmoid'),
            # keras arguments following
            validation_split = 0.25,
            callbacks = list(
              keras::callback_early_stopping(monitor = "val_accuracy", mode = "auto", 
                                             patience = 5, restore_best_weights = TRUE)
            ),
            epochs = 30)



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



# Nice code found on stackoverlow
# tune_model <- train(x, y, 
#                     method = "mlpKerasDropout",
#                     preProc = c('center', 'scale', 'spatialSign'),
#                     trControl = trainControl(search = 'random', classProbs = T, 
#                                              summaryFunction = mnLogLoss, allowParallel = TRUE),
#                     metric = 'logLoss',
#                     tuneLength = 20, 
#                     # keras arguments following
#                     validation_split = 0.25,
#                     callbacks = list(
#                       keras::callback_early_stopping(monitor = "val_loss", mode = "auto", 
#                                                      patience = 20, restore_best_weights = TRUE)
#                     ),
#                     epochs = 500)


set.seed(2)
training <- twoClassSim(50, linearVars = 2)
testing <- twoClassSim(500, linearVars = 2)
trainX <- training[, -ncol(training)]
trainY <- training$Class

rec_cls <- recipe(Class ~ ., data = training) %>%
  step_center(all_predictors()) %>%
  step_scale(all_predictors())

cctrl1 <- trainControl(method = "cv", number = 3, returnResamp = "all",
                       classProbs = TRUE, summaryFunction = twoClassSummary)
cctrl2 <- trainControl(method = "LOOCV")
cctrl3 <- trainControl(method = "none",
                       classProbs = TRUE, summaryFunction = twoClassSummary)
cctrlR <- trainControl(method = "cv", number = 3, returnResamp = "all", search = "random")

set.seed(849)
test_class_cv_model <- train(trainX, trainY, 
                             method = "mlpKerasDropout", 
                             trControl = cctrl3,
                             preProc = c("center", "scale"),
                             tuneLength = 2,
                             verbose = 0,
                             epochs = 10)


test_class_none_model <- train(trainX, trainY, 
                               method = "mlpKerasDropout", 
                               trControl = cctrl3,
                               tuneLength = 1,
                               metric = "ROC", 
                               preProc = c("center", "scale"),
                               verbose = 0,
                               epochs = 10)



y_caret_train <- y_caret_train[, 'V2']
y_caret_train <- factor(y_caret_train)
levels(y_caret_train) <- c('No', 'Yes')

rctrlR <- trainControl(method = "cv", number = 3, 
                       returnResamp = "all",
                       classProbs = TRUE, summaryFunction = multiClassSummary)


test_class_none_model <- train(x = x_caret_train, y = y_caret_train, 
                               method = "mlpKerasDropout", 
                               trControl = rctrlR,
                               metric = "accuracy", 
                               size = c(50, 50, 50),
                               dropout = c(0.15, 0.15, .15),
                               batch_size = c(5),
                               activation = c('relu', 'relu', 'relu', 'sigmoid'),
                               optimizer = 'adam',
                               loss = 'binary_crossentropy',
                               callbacks = list(
                                 keras::callback_early_stopping(monitor = "val_accuracy", min_delta = 0.0001,
                                                    patience = 5, restore_best_weights = TRUE)
                               ),
                               epochs = 30)
