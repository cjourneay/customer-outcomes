
########## get libraries

library(caret)
library(gbm)
library(plyr)
library(aCRM)
library(doParallel)
library(xgboost)
library(DMwR)
library(kernlab)

######### set up the working directory

setwd("C:/Users/chris/Desktop/Machine Learning/Labs/Coursework 2/Model Testing 5-4-2019")

#setwd("/home/cjour001/ML-CSWK2/Test")

rm(list = ls())

######################## start the cluster for computation #######################################################

cl <- makeCluster(detectCores()*0.5)
registerDoParallel(cl)

######################################################################################################
# Function that yields accuracy, kappa statistic, AUC with probabilities, sensitivity and specificity
######################################################################################################

# fOR ACCURACY, kAPPA, AUC and SENSITIVITY AND SPECIFICITY:
fiveStats <- function(...) c(twoClassSummary(...),
                             defaultSummary(...))

####################### SVM ######################################

bagged_trees <- function(trainingdata, testdata){
  
  appetency.data <- trainingdata
  appetency.data$appetency <- as.factor(appetency.data$appetency)
  
  test_1 <- preProcess(testdata, method = "knnImpute")
  testdata <- predict(test_1, testdata)
  
  #Smote Transformation:
  app_smote_train <- SMOTE(appetency~., data=appetency.data)
  table(app_smote_train$appetency)
  
  #  Applying the Spatial Sign Transformation:
  app_smote_train_pre <- preProcess(app_smote_train,
                                    method=c("spatialSign"))
  
  app_smote_train_pre <- predict(app_smote_train_pre,app_smote_train)

  
  app_bgg_smote_ctrl <- trainControl(method="repeatedcv",
                                     repeats=3,
                                     classProbs=TRUE,
                                     summaryFunction=fiveStats)
  # Start Training:
  set.seed(123)
  app_bgg <- train(make.names(appetency)~.,
                                 data=app_smote_train_pre,
                                 method="treebag",
                                 nbagg=50,
                                 metric="ROC",
                                 trControl=app_bgg_smote_ctrl)
  
  app_bggPred <- predict(app_bgg, testdata)

  # CALCULATE THE AREA UNDER THE CURVE (TRIANGLE+TRAPEZOID)
  
  TPR <- app_bgg$results$Sens # Sensitivity
  FPR <- (1-(app_bgg$results$Spec)) # 1-Specificity
  TPZ_AUC <- (0.5*(TPR*FPR)) + (0.5*((TPR)+1)*(1-FPR))  # Area of Trangle + Area of Trapezoid = AUC
  
  return(list(Trapezoid = TPZ_AUC, Predictions = app_bggPred))
}

######################## Load & run the data set ##################################################


Training_Set <- read.csv("appetency_Step_2A_train_Factor_None_Transform_ntile.csv", header=TRUE, sep = "\t")
Test_Set <- read.csv("appetency_Step_2A_test_Factor_None_Transform_ntile.csv", header=TRUE, sep = "\t")
Appetency_Prediction <- bagged_trees(Training_Set, Test_Set)
#Transforms the predictions into the appriate format
BGG_Prediction <- Appetency_Prediction$Prediction 
levels(BGG_Prediction)[levels(BGG_Prediction)=="X.1"] <- "-1"
levels(BGG_Prediction)[levels(BGG_Prediction)=="X1"] <- "1"
BGG_Prediction <- data.frame(BGG_Prediction)
names(BGG_Prediction) <- "appetency"

write.csv(BGG_Prediction, file = "Team_Taka_Appetency_Prediction.csv", row.names = FALSE)

######################### Stop the cluster #####################################################

stopCluster(cl)

######################################################

save.image(file = 'Team_Taka_Appetency_Prediction.RData')
