######################### MLSDM COURSEWORK - 2 #################################### 

############### Team Taka #######################

#Team Lead - Chris Journeay
#Team Members - Fabio Caputo, Imanol Belausteguigoitia

###################### Environment Prep  #############################

###### Set Local Drive Working Directories 

setwd("C:/Users/chris/Desktop/Machine Learning/Labs/Coursework 2/Final Preprocessing 4-4-2019/Appetency Preprocessing")

##### Set Cluster Working Directories 

#setwd("/home/cjour001/ML-CSWK2")

# Clears the workspace

rm(list=ls())

# Loads Libraries

library(caret)
library(doParallel)
library(randomForest)
library(dplyr)
library(AMR)
library(forcats)
library(imputeMissings)
library(mice)
library(AppliedPredictiveModeling)
library(CORElearn)

##### start cluster

cl <- makeCluster((detectCores() * 0.5))
registerDoParallel(cl)

################## Function ##############################

relief_transform <- function(training, test){
  
  reliefValues <- attrEval(as.factor(training[,1]) ~ .,
                                 data = training[,-1],
                                 estimator = "ReliefFequalK",
                                 ReliefIterations = 50)

  relief_perm <- permuteRelief(x = training[,-1],
                                     y = as.factor(training[,1]),
                                     nperm=500,
                                     estimator = "ReliefFequalK",
                                     ReliefIterations = 50)
  
  # Plot the distributions created with the permutations
  library(lattice)
  histogram(~value|Predictor, data=relief_perm$permutations)
  
  # Extract the standard deviations from the mean
  relief_perm_std <- abs(relief_perm$standardized) # since the value can be negative, let's compute it in absolute value
  sort(relief_perm_std, decreasing = TRUE)
  
  # Wrap up everything
  options(scipen=999)
  relief_perm_std <- data.frame(variable=names(relief_perm_std),
                                      `observed relief value`=as.numeric(reliefValues),
                                      std_away_perm_distr=as.numeric(relief_perm_std))
  
  columns_to_remove <- relief_perm_std %>% filter(std_away_perm_distr>1.86)
  columns_to_remove <- columns_to_remove$variable
  ####get this as a list and trim the data sets accordingly
  # no variable is more than 2 standard deviations away from the mean of the distribution obtained with permutations
  # with t-student value of = 1.86 
  # any variable with more than 1.86 standard deviations away from the mean should be removed
  
  #save.image(file = 'Chi_Squared_test.RData')
  training <- training[, !(colnames(training) %in% columns_to_remove)]
  test <- test[, !(colnames(test) %in% columns_to_remove)]
    return(list(relief_train = training, relief_test = test))
}


################# run on the appetency data

####### set 1
#loads the training and test data
appetency_1_train <- read.csv("appetency_Step_2A_train_Factor_None_Transform_ntile.csv", na.strings="", header=TRUE, sep="\t")
appetency_1_test <- read.csv("appetency_Step_2A_test_Factor_None_Transform_ntile.csv", na.strings="", header=TRUE, sep="\t")
#passes the data set to the Relief function
relief_results <- relief_transform(appetency_1_train, appetency_1_test)
#writes the modified training and test results to file
write.table(relief_results$relief_train, sep="\t", na = "" ,  file = "appetency_Step_3_relief_train_None_ntile.csv", row.names = FALSE)
write.table(relief_results$relief_test, sep="\t", na = "",  file = "appetency_Step_3_relief_test_None_ntile.csv", row.names = FALSE)
#releases the data from memory
rm(appetency_1_train, appetency_1_test, relief_results)

###### set 2
appetency_2_train <- read.csv("appetency_Step_2A_train_Factor_Mode_Transform_ntile.csv", na.strings="", header=TRUE, sep="\t")
appetency_2_test <- read.csv("appetency_Step_2A_test_Factor_Mode_Transform_ntile.csv", na.strings="", header=TRUE, sep="\t")

relief_results <- relief_transform(appetency_2_train, appetency_2_test)

write.table(relief_results$relief_train, sep="\t", na = "" ,  file = "appetency_Step_3_relief_train_Mode_ntile.csv", row.names = FALSE)
write.table(relief_results$relief_test, sep="\t", na = "",  file = "appetency_Step_3_relief_test_Mode_ntile.csv", row.names = FALSE)

rm(appetency_2_train, appetency_2_test, relief_results)

##### set 3
appetency_3_train <- read.csv("appetency_Step_2B_train_Factor_None_Transform_k_means.csv", na.strings="", header=TRUE, sep="\t")
appetency_3_test <- read.csv("appetency_Step_2B_test_Factor_None_Transform_k_means.csv", na.strings="", header=TRUE, sep="\t")

relief_results <- relief_transform(appetency_3_train, appetency_3_test)

write.table(relief_results$relief_train, sep="\t", na = "" ,  file = "appetency_Step_3_relief_train_None_k_means.csv", row.names = FALSE)
write.table(relief_results$relief_test, sep="\t", na = "",  file = "appetency_Step_3_relief_test_None_k_means.csv", row.names = FALSE)

rm(appetency_3_train, appetency_3_test, relief_results)

##### set 4
appetency_4_train <- read.csv("appetency_Step_2B_train_Factor_Mode_Transform_k_means.csv", na.strings="", header=TRUE, sep="\t")
appetency_4_test <- read.csv("appetency_Step_2B_test_Factor_Mode_Transform_k_means.csv", na.strings="", header=TRUE, sep="\t")

relief_results <- relief_transform(appetency_4_train, appetency_4_test)

write.table(relief_results$relief_train, sep="\t", na = "" ,  file = "appetency_Step_3_relief_train_Mode_k_means.csv", row.names = FALSE)
write.table(relief_results$relief_test, sep="\t", na = "",  file = "appetency_Step_3_relief_test_Mode_k_means.csv", row.names = FALSE)

rm(appetency_4_train, appetency_4_test, relief_results)

# stops the cluster

stopCluster(cl)

# save image file

save.image(file = "Appetency_Step_3_Relief.RData")