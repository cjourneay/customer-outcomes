######################### MLSDM COURSEWORK - 2 #################################### 

############### Team Taka #######################

#Team Lead - Chris Journeay
#Team Members - Fabio Caputo, Imanol Belausteguigoitia

###################### Environment Prep  #############################

###### Set Local Drive Working Directories 

setwd("C:/Users/chris/Desktop/Machine Learning/Labs/Coursework 2/Final Preprocessing 4-4-2019/Churn Preprocessing")

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
  
  columns_to_remove <- relief_perm_std %>% filter(std_away_perm_distr>1.68)
  columns_to_remove <- columns_to_remove$variable
  ####get this as a list and trim the data sets accordingly
  # no variable is more than 2 standard deviations away from the mean of the distribution obtained with permutations
  # with t-student value of = 1.68
  # any variable with more than 1.68 standard deviations away from the mean should be removed
  
  #save.image(file = 'Chi_Squared_test.RData')
  training <- training[, !(colnames(training) %in% columns_to_remove)]
  test <- test[, !(colnames(test) %in% columns_to_remove)]
    return(list(relief_train = training, relief_test = test, relief_ouput = relief_perm_std))
}


################# run on the churn data

####### set 1
#loads the training and test data
churn_1_train <- read.csv("churn_Step_2A_train_Factor_None_Transform_ntile.csv", na.strings="", header=TRUE, sep="\t")
churn_1_test <- read.csv("churn_Step_2A_test_Factor_None_Transform_ntile.csv", na.strings="", header=TRUE, sep="\t")
#passes the data set to the Relief function
relief_results <- relief_transform(churn_1_train, churn_1_test)
#writes the modified training and test results to file
write.table(relief_results$relief_train, sep="\t", na = "" ,  file = "churn_Step_3_relief_train_None_ntile.csv", row.names = FALSE)
write.table(relief_results$relief_test, sep="\t", na = "",  file = "churn_Step_3_relief_test_None_ntile.csv", row.names = FALSE)
write.table(relief_results$relief_output, sep="\t", na = "",  file = "churn_Step_3_relief_output_None_ntile.csv", row.names = FALSE)
#releases the data from memory
rm(churn_1_train, churn_1_test, relief_results)

###### set 2
churn_2_train <- read.csv("churn_Step_2A_train_Factor_Mode_Transform_ntile.csv", na.strings="", header=TRUE, sep="\t")
churn_2_test <- read.csv("churn_Step_2A_test_Factor_Mode_Transform_ntile.csv", na.strings="", header=TRUE, sep="\t")

relief_results <- relief_transform(churn_2_train, churn_2_test)

write.table(relief_results$relief_train, sep="\t", na = "" ,  file = "churn_Step_3_relief_train_Mode_ntile.csv", row.names = FALSE)
write.table(relief_results$relief_test, sep="\t", na = "",  file = "churn_Step_3_relief_test_Mode_ntile.csv", row.names = FALSE)
write.table(relief_results$relief_output, sep="\t", na = "",  file = "churn_Step_3_relief_output_Mode_ntile.csv", row.names = FALSE)
rm(churn_2_train, churn_2_test, relief_results)

##### set 3
churn_3_train <- read.csv("churn_Step_2B_train_Factor_None_Transform_k_means.csv", na.strings="", header=TRUE, sep="\t")
churn_3_test <- read.csv("churn_Step_2B_test_Factor_None_Transform_k_means.csv", na.strings="", header=TRUE, sep="\t")

relief_results <- relief_transform(churn_3_train, churn_3_test)

write.table(relief_results$relief_train, sep="\t", na = "" ,  file = "churn_Step_3_relief_train_None_k_means.csv", row.names = FALSE)
write.table(relief_results$relief_test, sep="\t", na = "",  file = "churn_Step_3_relief_test_None_k_means.csv", row.names = FALSE)
write.table(relief_results$relief_output, sep="\t", na = "",  file = "churn_Step_3_relief_output_None_k_means.csv", row.names = FALSE)

rm(churn_3_train, churn_3_test, relief_results)

##### set 4
churn_4_train <- read.csv("churn_Step_2B_train_Factor_Mode_Transform_k_means.csv", na.strings="", header=TRUE, sep="\t")
churn_4_test <- read.csv("churn_Step_2B_test_Factor_Mode_Transform_k_means.csv", na.strings="", header=TRUE, sep="\t")

relief_results <- relief_transform(churn_4_train, churn_4_test)

write.table(relief_results$relief_train, sep="\t", na = "" ,  file = "churn_Step_3_relief_train_Mode_k_means.csv", row.names = FALSE)
write.table(relief_results$relief_test, sep="\t", na = "",  file = "churn_Step_3_relief_test_Mode_k_means.csv", row.names = FALSE)
write.table(relief_results$relief_output, sep="\t", na = "",  file = "churn_Step_3_relief_output_Mode_k_means.csv", row.names = FALSE)

rm(churn_4_train, churn_4_test, relief_results)

# stops the cluster

stopCluster(cl)

# save image file

save.image(file = "Churn_Step_3_Relief.RData")