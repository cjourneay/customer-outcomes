
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

#setwd("Set WD Here")

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

####################### Random Forest ######################################

rand_forest <- function(trainingdata, testdata){
  
  upselling.data <- trainingdata
  upselling.data$upselling <- as.factor(upselling.data$upselling)
  
  test_1 <- preProcess(testdata, method = "knnImpute")
  testdata <- predict(test_1, testdata)
  
  #Smote Transformation:
  ups_smote_train <- SMOTE(upselling~., data=upselling.data)
  table(ups_smote_train$upselling)
  
  #  Applying the Spatial Sign Transformation:
  ups_smote_train_pre <- preProcess(ups_smote_train,
                                    method=c("spatialSign"))
  
  ups_smote_train_pre <- predict(ups_smote_train_pre,ups_smote_train)
  
  
  ups_rf_smote_ctrl <- trainControl(method="repeatedcv",
                                     repeats=3,
                                     classProbs=TRUE,
                                     summaryFunction=fiveStats)
  # Start Training:
  set.seed(123)
  ups_rf <- train(make.names(upselling)~.,
                   data=ups_smote_train_pre,
                   method="treebag",
                   nbagg=50,
                   metric="ROC",
                   trControl=ups_rf_smote_ctrl)
  
  ups_rfPred <- predict(ups_rf, testdata)
  
  # CALCULATE THE AREA UNDER THE CURVE (TRIANGLE+TRAPEZOID)
  
  TPR <- ups_rf$results$Sens # Sensitivity
  FPR <- (1-(ups_rf$results$Spec)) # 1-Specificity
  TPZ_AUC <- (0.5*(TPR*FPR)) + (0.5*((TPR)+1)*(1-FPR))  # Area of Trangle + Area of Trapezoid = AUC
  
  return(list(Trapezoid = TPZ_AUC, Predictions = ups_rfPred))

}

######################## Load & run the data set ##################################################


Training_Set <- read.csv("upselling_Step_2A_train_Factor_Mode_Transform_ntile.csv", sep='\t')
Test_Set <- read.csv("upselling_Step_2A_test_Factor_Mode_Transform_ntile.csv", sep='\t')

Upselling_Prediction <- rand_forest(Training_Set, Test_Set)
#Transforms the predictions into the appriate format
RF_Prediction <- Upselling_Prediction$Prediction 
levels(RF_Prediction)[levels(RF_Prediction)=="X.1"] <- "-1"
levels(RF_Prediction)[levels(RF_Prediction)=="X1"] <- "1"
RF_Prediction <- data.frame(RF_Prediction)
names(RF_Prediction) <- "upselling"

write.csv(RF_Prediction, file = "Team_Taka_Upselling_Prediction.csv", row.names = FALSE)

######################### Stop the cluster #####################################################

stopCluster(cl)

######################################################

save.image(file = 'Team_Taka_Upselling_Prediction.RData')
