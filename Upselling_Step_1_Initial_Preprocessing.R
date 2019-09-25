######################### MLSDM COURSEWORK - 2 #################################### 

############### Team Taka #######################

#Team Lead - Chris Journeay
#Team Members - Fabio Caputo, Imanol Belausteguigoitia

###################### Environment Prep  #############################

###### Set Local Drive Working Directories 

setwd("C:/Users/chris/Desktop/Machine Learning/Labs/Coursework 2/Final Preprocessing 4-4-2019/Upselling Preprocessing")

#load('Step_1_Initial_Preprocessing.RData')

##### Set Cluster Working Directories 

#setwd("/home/cjour001/ML-CSWK2")

# Clears the workspace
rm(list=ls())

# Loads Libraries

library(caret)
library(doParallel)
library(randomForest)
library(dplyr)
library(e1071)
library(forcats)
library(imputeMissings)
library(mlr)


# Imports the training data and targets
# test data is also imported so transformations to the training set can 
# mirrored on the test data 

trainingdata <- read.csv("train_X.csv", na.strings="", header=FALSE, sep="\t")
targets <- read.csv("train_Y.csv", na.strings="", header=TRUE, sep="\t")
testdata <- read.csv("test_X.csv", na.strings="", header=FALSE, sep="\t")

# imports the targets
targets <- read.csv("train_Y.csv", na.strings="", header=TRUE, sep="\t")
targets <- data.frame(apply(targets, 2, as.factor))

##### start cluster

cl <- makeCluster((detectCores() * 0.5))
registerDoParallel(cl)

############# STEP 1 - Remove columns based on data missing values ##########

# Sets the threshold to 70% of of data
data_cutoff <- 70 

#initialises vectors 
non_empty_column_values <- vector()
non_empty_column_values_percent <- vector()
unique_column_values <- vector()

#iterates through columns and calculates the 
# 1. the number of non_empty values in the column
# 2. the non-empty values as a percent of the overall column
# 3. the unique content values
for (i in 1:ncol(trainingdata)){
  non_empty_column_values[i] <- sum(!is.na(trainingdata[,i]))
  non_empty_column_values_percent[i] <- (non_empty_column_values[i]/nrow(trainingdata))*100
  unique_column_values[i] <- length(unique(trainingdata[,i]))
}

# creates a descending sorted list of the non-empty value percentages
# importantly this sorting keeps the original index values which represent the column index (or name) 
sorted_by_non_empty <- sort(non_empty_column_values_percent, decreasing = TRUE, index.return = TRUE)
print(sorted_by_non_empty$x) #prints out the percentages
print(sorted_by_non_empty$ix)#prints out the corresponding column index value 

# generates the list of columns to keep (by column index)
columns_to_keep <- (sorted_by_non_empty$ix[1:length(sorted_by_non_empty[sorted_by_non_empty$x > data_cutoff])])
print(columns_to_keep)

# updates the training set
trainingdata <- subset(trainingdata, select = c(columns_to_keep))

# updates the test set
testdata <- subset(testdata, select = c(columns_to_keep))

print(dim(trainingdata))
print(dim(testdata))

##################### STEP 2 - Centre and Scale numeric data ##############################################

#separates the dataset into numbers and factors
trainingdata_numbers <- trainingdata[, sapply(trainingdata, class) != "factor"]
trainingdata_factors <- trainingdata[, sapply(trainingdata, class) == "factor"]

#centres and scales the numerics
preprocessed_trainingdata = preProcess(trainingdata_numbers, method = c("center", "scale"))
trainingdata_numbers <- predict(preprocessed_trainingdata, trainingdata_numbers)

#updates the main training set to reflect the udpated numeric values
trainingdata <- data.frame(trainingdata_numbers, trainingdata_factors)

#applies transformation on test data
testdata_numbers <- testdata[, sapply(testdata, class) != "factor"]
testdata_factors <- testdata[, sapply(testdata, class) == "factor"]

testdata_numbers <- predict(preprocessed_trainingdata, testdata_numbers)

testdata <- data.frame(testdata_numbers, testdata_factors)

################ STEP 3 - Remove columns with near-zero variance #####################

# Note : the nearZeroVar function implements this logic: remove a predictor if:
# a. The fraction of unique values over the sample size is low (say 10%)
# b. The ratio of the frequency of the most prevalent value to the frequency of the 2nd most prevalent value is large (say 20)

# when predictor should be removed, a vector of integers is returned that indicates which columns should be removed
columns_to_remove <- nearZeroVar(trainingdata) 

nzv_columns <- subset(trainingdata, select = c(columns_to_remove))

# filter out the variables with non near-zero variance:
trainingdata <- trainingdata[! names(trainingdata) %in% names(nzv_columns)]

#apply the same filter to the test data
testdata <- testdata[!names(testdata) %in% names(nzv_columns)]

#################### STEP 4 - Impute missing numerical data ###############################

#separates the dataset into numbers and factors
trainingdata_numbers <- trainingdata[, sapply(trainingdata, class) != "factor"]
trainingdata_factors <- trainingdata[, sapply(trainingdata, class) == "factor"]

# attaches the target to the numerical data for imputation
trainingdata_numbers <- cbind(upselling=targets$upselling, trainingdata_numbers)

#imputes the training data 
trainingdata_numbers_impute <- mlr::impute(trainingdata_numbers, target = "upselling", classes = list(numeric = imputeNormal()))
trainingdata_numbers <- trainingdata_numbers_impute$data
trainingdata <- data.frame(trainingdata_numbers, trainingdata_factors)
#removes the target column
trainingdata <- trainingdata[,-1]

#imputes the testing data based on the coefficients from the training set
testdata_numbers <- testdata[, sapply(testdata, class) != "factor"]
testdata_factors <- testdata[, sapply(testdata, class) == "factor"]

testdata_numbers_impute <- reimpute(testdata_numbers, trainingdata_numbers_impute$desc)
testdata_numbers <- testdata_numbers_impute
testdata <- data.frame(testdata_numbers, testdata_factors)

################### STEP 5 - Address Skewness  ############################################################

#separates the dataset into numbers and factors
trainingdata_numbers <- trainingdata[, sapply(trainingdata, class) != "factor"]
trainingdata_factors <- trainingdata[, sapply(trainingdata, class) == "factor"]

fix_skewness <- preProcess(trainingdata_numbers, method = "YeoJohnson")
training_fix_skewness <- predict(fix_skewness, trainingdata_numbers)

trainingdata_numbers <- training_fix_skewness

#updates the main training set to reflect the udpated numeric values
trainingdata <- data.frame(trainingdata_numbers, trainingdata_factors)

#applies transformation on test data
testdata_numbers <- testdata[, sapply(testdata, class) != "factor"]
testdata_factors <- testdata[, sapply(testdata, class) == "factor"]

testdata_numbers <- predict(fix_skewness, testdata_numbers)

testdata <- data.frame(testdata_numbers, testdata_factors)

################### STEP 6A - Impute factors replacing N/A with "None" #########################

#separates the dataset into numbers and factors
trainingdata_numbers <- trainingdata[, sapply(trainingdata, class) != "factor"]
trainingdata_factors <- trainingdata[, sapply(trainingdata, class) == "factor"]

#imputes missing data for factors
trainingdata_None_Transform <- trainingdata_factors

for (i in 1:ncol(trainingdata_None_Transform)){
  levels <- levels(trainingdata_None_Transform[,i])
  levels[length(levels) + 1] <- "None"
  #refactor column to include "None" as a factor level and replace NA with "None"
  trainingdata_None_Transform[,i] <- factor(trainingdata_None_Transform[,i], levels = levels)
  trainingdata_None_Transform[,i][is.na(trainingdata_None_Transform[,i])] <- "None"
}

#updates the main training set to reflect the udpated factors values
trainingdata_Factor_None_Transform <- data.frame(trainingdata_numbers, trainingdata_None_Transform)

# Repeats the same transformation in the test data 
testdata_numbers <- testdata[, sapply(testdata, class) != "factor"]
testdata_factors <- testdata[, sapply(testdata, class) == "factor"]

testdata_None_Transform <- testdata_factors

for (i in 1:ncol(testdata_None_Transform)){
  levels <- levels(testdata_None_Transform[,i])
  levels[length(levels) + 1] <- "None"
  #refactor column to include "None" as a factor level and replace NA with "None"
  testdata_None_Transform[,i] <- factor(testdata_None_Transform[,i], levels = levels)
  testdata_None_Transform[,i][is.na(testdata_None_Transform[,i])] <- "None"
}

#updates the main test set to reflect the udpated factors values
testdata_Factor_None_Transform <- data.frame(testdata_numbers, testdata_None_Transform)

################## STEP 6B - Impute factors with N/A with mode ################################

#separates the dataset into numbers and factors
trainingdata_numbers <- trainingdata[, sapply(trainingdata, class) != "factor"]
trainingdata_factors <- trainingdata[, sapply(trainingdata, class) == "factor"]

# creates the output values from the training set that we apply to the training set

#runs the imputation on the training data set factors 
trainingdata_mode_impute_factors <- imputeMissings::impute(trainingdata_factors)
trainingdata_Factor_Mode_Transform <- data.frame(trainingdata_numbers,trainingdata_mode_impute_factors )

#performs the same operation on the test data
testdata_numbers <- testdata[, sapply(testdata, class) != "factor"]
testdata_factors <- testdata[, sapply(testdata, class) == "factor"]

values <- imputeMissings::compute(trainingdata_factors)
testdata_factors_mode_impute <- imputeMissings::impute(testdata_factors, object = values)

testdata_Factor_Mode_Transform <- data.frame(testdata_numbers, testdata_factors_mode_impute)

######################## Save Output #########################################################

write.table(trainingdata_Factor_None_Transform, sep="\t", na = "" ,  file = "Step_1_train_Factor_None_Transform.csv", row.names = FALSE)
write.table(testdata_Factor_None_Transform, sep="\t", na = "",  file = "Step_1_test_Factor_None_Transform.csv", row.names = FALSE)

write.table(trainingdata_Factor_Mode_Transform, sep="\t", na = "",file = "Step_1_train_Factor_Mode_Transform.csv", row.names = FALSE)
write.table(testdata_Factor_Mode_Transform, sep="\t", na = "",file = "Step_1_test_Factor_Mode_Transform.csv", row.names = FALSE)

save.image(file = 'Upselling_Step_1_Initial_Preprocessing.RData')

# stops the cluster
stopCluster(cl)
