
#===========================================================================================================================
# MLSDM COURSEWORK, Churn Modelling
#===========================================================================================================================

#clears the workspace
rm(list=ls())

# Import libraries
library(caret)
library(data.table)
library(dplyr)
library(doParallel)
library(DMwR)

setwd("PLS ENTER THE WORKING DIRECTORY WHERE THE .csv FILES BELOW ARE STORED")

# Check the working directory:
getwd()

# Training set:
chn_trn_mod_ntl_1 <- read.csv("churn_Step_2A_train_Factor_Mode_Transform_ntile.csv", sep='\t')
# Test set:
chn_tst_mod_ntl_1 <- read.csv("churn_Step_2A_test_Factor_Mode_Transform_ntile.csv", sep='\t')

#####################################################################
############# centre and scale the ordinal variables  ###############
#####################################################################

# Training set:
colnames(chn_trn_mod_ntl_1)

scale_ordinal <- preProcess(chn_trn_mod_ntl_1[,2:25], method=c("center","scale"))
chn_trn_mod_ntl_1 <- predict(scale_ordinal,chn_trn_mod_ntl_1)

View(chn_trn_mod_ntl_1)

# Test set:
colnames(chn_tst_mod_ntl_1)
scale_ordinal <- preProcess(chn_tst_mod_ntl_1[,1:24], method=c("center","scale"))
chn_tst_mod_ntl_1 <- predict(scale_ordinal,chn_tst_mod_ntl_1)

######################################################################
########################## PRINCIPAL COMPONENT ANALYSIS  #############
######################################################################

## Training set:
pr.out = prcomp(chn_trn_mod_ntl_1[,-1])

pr.out$rotation = -pr.out$rotation
pr.var = pr.out$sdev^2
pr.out$x = -pr.out$x

pr.out$x

biplot(pr.out,scale=0)

pve = pr.var/sum(pr.var)
plot(pve, xlab='',ylab='',ylim=c(0,1), type='b')
plot(cumsum(pve),xlab='',ylab='',ylim=c(0,1), type='b', col="red")

# check cumulative variance explained
cumsum(pve) # first 30 components explain 85% of the variance in the data ; the first 24 explain ~80%

# Implement the PCA transformation on the original data set
chn_pca1 <- preProcess(chn_trn_mod_ntl_1,method = c("pca"))
chn_pca <- predict(chn_pca1, chn_trn_mod_ntl_1)

chn_pca_train = chn_pca[, 1:31]

ncol(chn_pca)
ncol(chn_pca_train)
ncol(chn_pca)-ncol(chn_pca_train) # 16 variables have been dropped

View(chn_pca_train)

## PCA Test set:
chn_pca_test_1 <- preProcess(chn_tst_mod_ntl_1,method = c("pca"))
chn_pca_test <- predict(chn_pca_test_1, chn_tst_mod_ntl_1)

chn_pca_test_85 = chn_pca_test[, 1:30]

#####################################################################################
###################### Implement SMOTE Transformation to balance the classes:
#####################################################################################

table(chn_trn_mod_ntl_1$churn)
chn_smote_train <- SMOTE(churn~., data=chn_trn_mod_ntl_1)
table(chn_smote_train$churn)

# SMOTE WITH PCA:
table(chn_pca_train$churn)
chn_pca_smote_train <- SMOTE(churn~., data=chn_pca_train)
table(chn_pca_smote_train$churn)

####################### Applying the Spatial Sign Transformation:

# Training set:
chn_smote_train_pre <- preProcess(chn_smote_train[,-1],method=c("spatialSign"))
chn_smote_train_pre <- predict(chn_smote_train_pre,chn_smote_train)
View(chn_smote_train_pre)

# Test set:
chn_tst_sps_mod_ntl_1 <- preProcess(chn_tst_mod_ntl_1,method=c("spatialSign"))
chn_tst_sps_mod_ntl_1 <- predict(chn_tst_sps_mod_ntl_1,chn_tst_mod_ntl_1)
View(chn_tst_sps_mod_ntl_1)

# PCA, Training set:
chn_pca_smt_sps_train_pre <- preProcess(chn_pca_smote_train[,-1],method=c("spatialSign"))
chn_pca_smt_sps_train_pre <- predict(chn_pca_smt_sps_train_pre,chn_pca_smote_train)
View(chn_pca_smt_sps_train_pre)

## PCA, Test Set:
chn_pca_sps_test <- preProcess(chn_pca_test, method=c("spatialSign"))
chn_pca_sps_test <- predict(chn_pca_sps_test,chn_pca_test)
View(chn_pca_sps_test)

## Combo PCA and data set Training set:
chn_pcaplusBGG = cbind(chn_smote_train_pre ,chn_pca_smt_sps_train_pre[,2:31])
colnames(chn_pcaplusBGG)

## Combo PCA and data set Test set:
chn_pcaplusmodel_test = cbind(chn_tst_sps_mod_ntl_1 ,chn_pca_sps_test[,1:30])
colnames(chn_pcaplusmodel_test)


# fOR ACCURACY, kAPPA, AUC and SENSITIVITY AND SPECIFICITY:
fiveStats <- function(...) c(twoClassSummary(...),
                             defaultSummary(...))



# Function for calculating AUC (TRAPEZOID METHOD):
trapezoid_area <- function(model){
best <<- which.max(model$results$Sens) # Select the best model
TPR <- model$results$Sens[best] # Sensitivity
FPR <- (1-(model$results$Spec)[best]) # 1-Specificity
TPZ_AUC <<- (0.5*(TPR*FPR)) + (0.5*((TPR)+1)*(1-FPR))}  # Area of Trangle + Area of Trapezoid = AUC

# Function for calculating AUC (TRAPEZOID METHOD):
trapezoid_area_Kappa <- function(model){
  best <<- which.max(model$results$Kappa)
  TPR <- model$results$Sens[best] # Sensitivity
  FPR <- (1-(model$results$Spec)[best]) # 1-Specificity
  TPZ_AUC_KPP <<- (0.5*(TPR*FPR)) + (0.5*((TPR)+1)*(1-FPR))}


###################################################################################################################
############ MODEL SELECTED FOR PREDICTIONS ON TEST : EXTREME GRADIENT BOOSTING AND PCA TOGETHER ##################
###################################################################################################################

chn_pcaplusXGB <- chn_pcaplusBGG

#calculates the event rate of churn
chn_event_rate <- nrow(chn_pcaplusXGB[chn_pcaplusXGB$churn == "1",])/ nrow(chn_pcaplusXGB)


# Train COntrol
chn_xgb_smote_ctrl <- trainControl(method="repeatedcv",
                                   repeats=3,
                                   classProbs=TRUE,
                                   summaryFunction=fiveStats,
                                   allowParallel = TRUE)

# Grid
chn_xgbGrid <- expand.grid(nrounds = 100,
                           max_depth = c(5,7,12,15),
                           eta = c(0.01,0.2,0.3,0.5),
                           gamma = 0, # ?
                           colsample_bytree = c(0.3,0.6), 
                           min_child_weight = (1/sqrt(chn_event_rate)),
                           subsample =1)

# Model
cl <- makeCluster(detectCores())
registerDoParallel(cl)
#sets the random variable used to calculate splits
set.seed(358)
#runs the model (still running..)
chn_pcaPLUSxgbFit <- train(make.names(churn) ~., 
                           data = chn_pcaplusXGB,
                           method = "xgbTree",
                           trControl = chn_xgb_smote_ctrl,
                           verbose = TRUE,
                           tuneGrid = chn_xgbGrid)
stopCluster(cl)

# The area under the curve:
trapezoid_area(chn_pcaPLUSxgbFit)
xgbPLUSpca_TPZ_AUC <- TPZ_AUC # 0.9925433

# Function for calculating AUC (TRAPEZOID METHOD):
trapezoid_area_Kappa(chn_pcaPLUSxgbFit)
xgbPLUSpca_TPZ_KPP_AUC <- TPZ_AUC_KPP # 0.9925433


####################################################################################################
##### PREDICTIONS: Best Model : extreme gradient boosting trees PLUS Principal Components:
####################################################################################################

# The test set for deploying the best model:
View(chn_pcaplusmodel_test)

# Deplying the model on the test set:
chn_bst_mdl <- predict(chn_pcaPLUSxgbFit,chn_pcaplusmodel_test)
chn_bst_mdl2 <- as.data.frame(chn_bst_mdl)

# Adjusting the labels' names (altered as a result of the "make.names" function)
chn_bst_mdl3 <- chn_bst_mdl2 %>% mutate(Churn=if_else(chn_bst_mdl=="X1",1,-1)) %>% select(-chn_bst_mdl)
table(chn_bst_mdl3)

#   -1     1 
# 13823  3176

write.csv(chn_bst_mdl3,"chn_bst_predictions.csv")

