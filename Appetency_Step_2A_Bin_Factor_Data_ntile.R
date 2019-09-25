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

##### start cluster

cl <- makeCluster((detectCores() * 0.5))
registerDoParallel(cl)


# Loads Libraries

library(caret)
library(doParallel)
library(dplyr)
library(e1071)

######################## Load Data  #########################################################

#imports files created in Step 1 of the training

trainingdata_Factor_None_Transform <- read.csv("Step_1_train_Factor_None_Transform.csv", na.strings="", header=TRUE, sep="\t")
testdata_Factor_None_Transform <- read.csv("Step_1_test_Factor_None_Transform.csv", na.strings="", header=TRUE, sep="\t")

trainingdata_Factor_Mode_Transform <- read.csv("Step_1_train_Factor_Mode_Transform.csv", na.strings="", header=TRUE, sep="\t")
testdata_Factor_Mode_Transform <- read.csv("Step_1_test_Factor_Mode_Transform.csv", na.strings="", header=TRUE, sep="\t")

# imports the targets
targets <- read.csv("train_Y.csv", na.strings="", header=TRUE, sep="\t")
targets <- data.frame(apply(targets, 2, as.factor))

################ Function to calculate reponse rates of factor levels ##########################

response_rate <- function(iterator, whole_data_frame){
  assessment_data_frame <- data.frame(whole_data_frame[,1], whole_data_frame[,iterator])
  names(assessment_data_frame) <- c("target", "level_names")
  response_rate_summary <- 
    assessment_data_frame %>% dplyr::group_by(level_names) %>%
    dplyr::summarise(total = length(level_names),
              positive = sum(target == 1), 
              negative = sum(target != 1),
              positiverate = positive/total,
              negativerate = 1-positiverate)
  return(response_rate_summary)
}

###################### Zero Repsonse Check

zero_response_check <- function(training){
  training_factors <- training[, sapply(training, class) == "factor"]
  training_factors_names <- names(training_factors)
  zero_response_columns <- vector()
  
  for (i in 2:ncol(training_factors)){
    responses <- data.frame(response_rate(i, training_factors))
    if (sum(responses$positive) == 0){
      zero_response_columns <- append(zero_response_columns, training_factors_names[i])
    }
  }
  return(zero_response_columns)
}

############# Bins factors by quartile based on response rate to the target ##########################################

n_tile_transform <- function(target_name, training, test){
  
  #separates the datasets into numbers and factors

  training_numbers <- training[, sapply(training, class) != "factor"]
  training_factors <- training[, sapply(training, class) == "factor"]

  test_numbers <- test[, sapply(test, class) != "factor"]
  test_factors <- test[, sapply(test, class) == "factor"]

  training_factors <- cbind(targets[target_name], training_factors)
  training_factor_names <- colnames(training_factors)
  test_factor_names <- colnames(test_factors)

  # performs a zero response check 
  # removes any factor columns with zero response to the target
  
  zero_response <- zero_response_check(training_factors)
  if (length(zero_response)>0){
      training_factors <- training_factors[,!(names(training_factors)%in% zero_response)]
  } 
  
  #starts the process of binning the data
  training_new_cols <- vector()
  training_new_cols[1] <- target_name
  test_new_cols <- vector()
  
  #frames to fill with values
  skew_transformation <- data.frame(matrix(nrow=ncol(training_factors)-1, ncol=3))
  names(skew_transformation) <- c("column_name", "original_skewness", "Box_Cox_skewness")
  
  #iterates through the columns
  for (i in 2:ncol(training_factors)){
    #creates a dataframe with a frequency table for factor levels for each positive outcome
    responses <- data.frame(response_rate(i, training_factors))
    #sets up box-cox transform for the skewness
    preProcValues <- preProcess(responses[5], method = c("BoxCox", "center", "scale"))
    unskew_responses <- predict(preProcValues,responses)
    
    #splits resulting valaues into 4 quartiles 
    unskew_responses$ntile <- ntile(unskew_responses$positiverate, 4)
    
    #merges the quartile values into the factors data frame both on training and test data
    training_factors$ntile <- unskew_responses[match(training_factors[,i], unskew_responses$level_name), "ntile"]
    test_factors$ntile <- unskew_responses[match(test_factors[,(i-1)], unskew_responses$level_name), "ntile"]
    
    #replaces NA values (ones that do not appear in the positive values) with 5
    training_factors <- tidyr::replace_na(training_factors, list(ntile=5))
    test_factors <- tidyr::replace_na(test_factors, list(ntile=5))
    
    #renames the column to include the original column reference - repeated on the test set
    names(training_factors)[ncol(training_factors)] <- paste0('ntile_', training_factor_names[i])
    training_new_cols[i] <- paste0('ntile_', training_factor_names[i])
    
    names(test_factors)[ncol(test_factors)] <- paste0('ntile_', test_factor_names[(i-1)])
    test_new_cols[i-1] <- paste0('ntile_', test_factor_names[(i-1)])
    
    #captures the values of the skew transformations for reporting
    skew_transformation_new_row <- list(training_factor_names[i], skewness(responses$positiverate),skewness(unskew_responses$positiverate))
    names(skew_transformation_new_row) <- c("column_name", "original_skewness", "Box_Cox_skewness")
    skew_transformation <- bind_rows(skew_transformation, skew_transformation_new_row)
    skew_transformation_new_row <- NULL
  }
  
  #drops the original columns from the ntile results
  training_factors <- training_factors[training_new_cols]
  test_factors <- test_factors[test_new_cols]
  
  #converts any numerics to factors
  training_factors <- data.frame(apply(training_factors, 2, as.factor))
  test_factors <- data.frame(apply(test_factors, 2, as.factor))
  
  #re-assembles the training and test data sets with the modified factors
  training <- data.frame(training_factors, training_numbers)
  test <- data.frame(test_factors, test_numbers)
  
  #returns the new training and test sets 
  return(list(ntile_training = training, ntile_test = test))
  
}

####################### BIN Factors for apptetency ############################################################# 

####### run the transformation on the NA-> None factor data set

ntile_results <- n_tile_transform("appetency", trainingdata_Factor_None_Transform, testdata_Factor_None_Transform)

write.table(ntile_results$ntile_training, sep="\t", na = "" ,  file = "appetency_Step_2A_train_Factor_None_Transform_ntile.csv", row.names = FALSE)
write.table(ntile_results$ntile_test, sep="\t", na = "",  file = "appetency_Step_2A_test_Factor_None_Transform_ntile.csv", row.names = FALSE)

####### run the transformation on the NA Mode imputed factor data set

ntile_results <- n_tile_transform("appetency", trainingdata_Factor_Mode_Transform, testdata_Factor_Mode_Transform)

write.table(ntile_results$ntile_training, sep="\t", na = "" ,  file = "appetency_Step_2A_train_Factor_Mode_Transform_ntile.csv", row.names = FALSE)
write.table(ntile_results$ntile_test, sep="\t", na = "",  file = "appetency_Step_2A_test_Factor_Mode_Transform_ntile.csv", row.names = FALSE)

# stops the cluster

stopCluster(cl)

################# save output 

save.image(file = "Appetency_Step_2A_factor_ntile_bin.RData")
