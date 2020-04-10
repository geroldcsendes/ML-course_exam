# This script reads in raw data and munges it
# Outputs are (interim) data
library(data.table)

# Read in data
data_train <- fread('data/train.csv')
data_test <- fread('data/test.csv')

# take a basic look at data
dim(data_train)
dim(data_test)

skim(data_train)

# get rid of non-predictive vars
data_train[, c( 'article_id'):= NULL]
# for test article_is needed

# select factor vars
factor_vars <- colnames(data_train)
factor_vars <- factor_vars[grepl('is', factor_vars)]

# convert to factor
data_train[, (factor_vars) := lapply(.SD, factor), .SDcols = factor_vars]
# for binary prediction, the target variable must be a factor with levels
levels(data_train$is_popular) <- c("No", "Yes")


# convert test to factor as well
# is_popular is not in test
factor_vars_test <- setdiff(factor_vars, 'is_popular')
data_test[, (factor_vars_test) := lapply(.SD, factor), .SDcols = factor_vars_test]
# str(data_test)

############## Scale ##############
# train data
preProcValues  <- preProcess(data_train, method=c('center','scale'))
trainTransformed <- predict(preProcValues, data_train)

# test data
# IMPORTANT: transform test via train transformation values
testTransformed <- predict(preProcValues, data_test)

############## Scale ##############
saveRDS(data_train, 'data/data_train.RDS')
saveRDS(data_test, 'data/data_test.RDS')
saveRDS(trainTransformed, 'data/trainTransformed.RDS')
saveRDS(testTransformed, 'data/testTransformed.RDS')

