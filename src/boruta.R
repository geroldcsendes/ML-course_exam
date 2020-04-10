library(data.table)
library(caret)
library(MLmetrics)
library(kableExtra)
library(skimr)
library(Boruta) # for finding relevant vars for LM model

# load in data
data_train <- readRDS('data/data_train.RDS')
data_test <- readRDS('data/data_test.RDS')

# Only use a subset of data, otherwise runs forever..
set.seed(1)
boruta_data<- data_train[sample(nrow(data_train), 3000), ]
set.seed(1)
boruta_train <- Boruta(
  is_popular ~.,
  data = boruta_data,
  doTrace = 3
)

# Take a look at the result
boruta_train # 38 features confirmed, 11 UNIMPORTANT, 9 TENTATIVE
saveRDS(boruta_train, 'data/boruta/boruta.RDS')
# boruta_decision <- boruta_train$finalDecision
# names(boruta_decision[boruta_decision == 'Tentative'])
# getSelectedAttributes(boruta_train, withTentative = F)

# get boruta df with descending importances
boruta_df <- attStats(boruta_train)
boruta_df <- boruta_df[order(-boruta_df$medianImp),]
write.csv(boruta_df, 'data/boruta/boruta_res.csv')


### Drop attributes based on boruta #### 
# Seems like day of publishing doesn' matter at all
# let's remove weekday vars
boruta_df$varname <- rownames(boruta_df)
rownames(boruta_df) <- NULL
boruta_vars <- boruta_df$varname
rel_vars <- boruta_vars[!grepl('weekday_is', boruta_vars)]
# filter out weekday cols
boruta_rel_df <- boruta_df[boruta_vars %in% rel_vars, ]

# define different variables sets for train and test
# test: + article_is
rel_vars_train <- rel_vars
rel_vars_train <- c(rel_vars_train, 'is_popular')
rel_vars_test <- c(rel_vars, 'article_id')


# filter datasets based on previous filters
boruta_train_filtered <- data_train[, ..rel_vars_train]
boruta_test_filtered <- data_test[, ..rel_vars_test]
# write_results
saveRDS(boruta_train_filtered, 'data/boruta/boruta_train.RDS')
saveRDS(boruta_test_filtered, 'data/boruta/boruta_test.RDS')


# Do the same on transformed data
trainTransformed <- readRDS('data/trainTransformed.RDS')
testTransformed <- readRDS('data/testTransformed.RDS')

boruta_trainTransformed <- trainTransformed[, ..rel_vars_train]
boruta_testTransformed <- testTransformed[, ..rel_vars_test]
# write results
saveRDS(boruta_trainTransformed, 'data/boruta/boruta_trainTransformed.RDS')
saveRDS(boruta_testTransformed, 'data/boruta/boruta_testTransformed.RDS')



# boruta_df_factor <- boruta_df[factor_vars,]
# boruta_df_factor[order(-boruta_df_factor$medianImp),]