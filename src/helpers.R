### Select top performants
top_performers <- function(model, model_params, criterion) {
  
  # extract results
  model_results <- model$results
  model_results <- model_results[order(- model_results$AUC),]
  model_results <- model_results[, model_params]
  
  # select results within 0.005% performance in AUC
  top_AUC <- max(model_results$AUC)
  AUC_min <- top_AUC - (top_AUC * criterion)
  print(AUC_min)
  
  model_results <- model_results[model_results[["AUC"]] >= AUC_min, ]
  
  return(model_results)
}

# recalcualte top rf models om whole train
estimate_top_rf_models <- function(tune_grid, data_train, data_test) {
  
  # get rid of AUC col
  tune_grid <- tune_grid[, setdiff(colnames(tune_grid), 'AUC')]
  
  # iterate over params
  for (row in seq(1, dim(tune_grid)[1])) {
    # extract params
    print(row)
    param_grid <- tune_grid[row, ] 
    print(param_grid)
    
    # Set other train control for final models
    train_control_finals <- trainControl(
      method = "none",
      classProbs = TRUE,
      summaryFunction = multiClassSummary # needed for AUC metric
    )
    
    # train model
    set.seed(1)
    rf_model_iter <- train(
      is_popular ~.,
      method = "ranger",
      data=data_train,
      tuneGrid = param_grid,
      trControl = train_control_finals,
      verbose = FALSE,
      metric = "AUC",
      importance = 'impurity'
    )
    
    # # save model
    param_cols <- colnames(param_grid)
    param_values <- as.character(as.vector(param_grid))
    modelname <- paste0(paste0(param_cols, param_values), collapse = '_')
    print(modelname)
    saveRDS(rf_model_iter, paste0('models/rf/', modelname, '.RDS'))
    
    # predict and save
    rf_iter_preds <- data_test[, 'article_id']
    rf_iter_preds$score  <- predict(rf_model_iter, newdata = data_test, type="prob")$Yes
    csv_name <- paste0('data/rf_preds/', modelname, '.csv')
    write.csv(rf_iter_preds, csv_name, row.names = F)
  }
}

# recalcualte top xgbm models om whole train
estimate_top_xgbm_models <- function(tune_grid, data_train, data_test) {
  
  # get rid of AUC col
  tune_grid <- tune_grid[, setdiff(colnames(tune_grid), c('model', 'AUC'))]
  
  # iterate over params
  for (row in seq(1, dim(tune_grid)[1])) {
    # extract params
    print(row)
    param_grid <- tune_grid[row, ] 
    print(param_grid)
    
    # Set other train control for final models
    train_control_finals <- trainControl(
      method = "none",
      classProbs = TRUE,
      summaryFunction = multiClassSummary # needed for AUC metric
    )
    
    # train model
    set.seed(1)
    xgbm_model_iter <- train(
      is_popular ~.,
      method = "xgbTree",
      data=data_train,
      tuneGrid = param_grid,
      trControl = train_control_finals,
      verbose = FALSE,
      metric = "AUC"
    )
    
    # # save model
    param_cols <- colnames(param_grid)
    param_values <- as.character(as.vector(param_grid))
    modelname <- paste0(paste0(param_cols, param_values), collapse = '_')
    print(modelname)
    saveRDS(xgbm_model_iter, paste0('models/xgbm_boruta/', modelname, '.RDS'))
    
    # predict and save
    xgbm_model_preds <- data_test[, 'article_id']
    xgbm_model_preds$score  <- predict(xgbm_model_iter, newdata = data_test, type="prob")$Yes
    csv_name <- paste0('data/xgbm_preds_boruta/', modelname, '.csv')
    write.csv(xgbm_model_preds, csv_name, row.names = F)
  }
}
