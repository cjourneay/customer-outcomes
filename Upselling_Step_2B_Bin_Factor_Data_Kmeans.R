######################### MLSDM COURSEWORK - 2 #################################### 

############### Team Taka #######################

#Team Lead - Chris Journeay
#Team Members - Fabio Caputo, Imanol Belausteguigoitia

###################### Environment Prep  #############################

###### Set Local Drive Working Directories 

setwd("C:/Users/chris/Desktop/Machine Learning/Labs/Coursework 2/Final Preprocessing 4-4-2019/Upselling Preprocessing")

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

###################### Zero Repsonse Check #############################

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

######################

k_means_transform <- function(target_name, data_source, training, test){
  
  #separates the datasets into numbers and factors
    training_numbers <- training[, sapply(training, class) != "factor"]
  training_factors <- training[, sapply(training, class) == "factor"]
  
  test_numbers <- test[, sapply(test, class) != "factor"]
  test_factors <- test[, sapply(test, class) == "factor"]
  
  training_factors <- cbind.data.frame(targets[target_name], training_factors)
  training_factor_names <- colnames(training_factors)
  test_factor_names <- colnames(test_factors)
  
  # performs a zero response check 
  # removes any factor columns with zero response to the target
  
  zero_response <- zero_response_check(training_factors)
  
  if (length(zero_response)>0){
    
    training_factors <- training_factors[,!(names(training_factors)%in% zero_response)]
  } 
  
  #Starts the binning process
  
  training_new_cols <- vector()
  training_new_cols[1] <- target_name
  test_new_cols <- vector()
  
  #sets parameters for managing the k-means process
  max_neighbours <- 10
  k_means <- data.frame(matrix(nrow = max_neighbours, ncol = (ncol(training_factors)-1)))
  names(k_means) <- names(training_factors[2:ncol(training_factors)])
  row.names(k_means) <- (1:max_neighbours)

  # iterate through the columns to run the k-means scenarios
  for (i in 2:ncol(training_factors)){
    set.seed(123)
    responses <- data.frame(response_rate(i, training_factors))
    #check to ensure that factors with < 10 levels can be processed by this method
    k_max <- 
        if(nrow(responses) < max_neighbours){nrow(responses)-1}
        # condition for edge case where there are a large number of zero responses   
        else if (nrow(responses)-nrow(responses[responses$positiverate == 0,])< max_neighbours)
          {nrow(responses)-nrow(responses[responses$positiverate == 0,])}
        else{max_neighbours}
    for (k in 1:k_max){
        kmeans_result <- kmeans(responses[,5:6], k)
        k_means[k,i-1] <- kmeans_result$tot.withinss}
    }
 
#saves the output from the testing as a file
write.csv(k_means, file=paste0(target_name, "_", data_source, "_k_means.csv"), row.names = FALSE)

#prints out the elbow graph for each column
for (i in 1:ncol(k_means)){
  #creates the output image file
  png(paste0(target_name, "_", data_source, "_", i, "_", colnames(k_means[i]), "_response_rate.png"))
  #plots the elbow graph
  plot(1:max_neighbours, k_means[,i], type = "b", 
       xlab = "Number of Clusters", 
       ylab = "Within groups sum of squares",
       main =  paste0(target_name, "_", data_source, "_", i, "_", colnames(k_means[i]))
  )
  #closes the file
  dev.off()
}

# chosen k values from the elbow graphs are input here
chosen_k = c(3,3,3,2,
             3,5,3,4,
             1,3,3,3,
             4,3,3,1,
             4,2,5,3,
             1,2,2,3)

#runs the optimal k-means one each column and creates a column with the cluster assignment
for (i in 2:ncol(training_factors)){
  set.seed(123)
  
    #####add in check to ensure that we do not run the k-means if factors < 4 
    if (nlevels(training_factors[,i]) >= 3){
      responses <- data.frame(response_rate(i, training_factors))
      optimal_k <- kmeans(responses[,5:6], chosen_k[i-1])
      responses$cluster <- optimal_k$cluster
  
      training_factors$cluster <- responses[match(training_factors[,i], responses$level_names), "cluster"]
      test_factors$cluster <- responses[match(test_factors[,i-1], responses$level_names), "cluster"]
      
      #renames the column to include the original column reference
      names(training_factors)[ncol(training_factors)] <- paste0("k-means_", training_factor_names[i])
      training_new_cols[i] <- paste0('k-means_', training_factor_names[i])
      #performs the same change to the test set                             
      names(test_factors)[ncol(test_factors)] <- paste0("k-means_", test_factor_names[(i-1)])
      test_new_cols[(i-1)] <- paste0('k-means_', test_factor_names[(i-1)])

  } 
  # if factors are less 4 then simply use the current factor values
    else {
      training_factors$cluster <- training_factors[,i]
      names(training_factors)[ncol(training_factors)] <- paste0("k-means_", training_factor_names[i])
      training_new_cols[i] <- paste0('k-means_', training_factor_names[i])
    
      test_factors$cluster <- test_factors[,i-1]
      names(test_factors)[ncol(test_factors)] <- paste0("k-means_", test_factor_names[(i-1)])
      test_new_cols[(i-1)] <- paste0('k-means_', test_factor_names[(i-1)])
        }
  }

#drops the original columns from the results
training_factors <- training_factors[training_new_cols]
test_factors <- test_factors[test_new_cols]

#converts any numerics to factors
training_factors <- data.frame(apply(training_factors, 2, as.factor))
test_factors <- data.frame(apply(test_factors, 2, as.factor))

#re-assembles the training and test data sets with the modified factors
training <- data.frame(training_factors, training_numbers)
test <- data.frame(test_factors, test_numbers)

return(list(k_means_training = training, k_means_test = test))

}

############################# run k-means on the upselling data sets

k_means_results <- k_means_transform("upselling", "NA_Transform", trainingdata_Factor_None_Transform, testdata_Factor_None_Transform)

write.table(k_means_results$k_means_training, sep="\t", na = "" ,  file = "upselling_Step_2B_train_Factor_None_Transform_k_means.csv", row.names = FALSE)
write.table(k_means_results$k_means_test, sep="\t", na = "",  file = "upselling_Step_2B_test_Factor_None_Transform_k_means.csv", row.names = FALSE)

####### run the transformation on the NA Mode imputed factor data set

k_means_results <- k_means_transform("upselling", "Mode_Transform", trainingdata_Factor_Mode_Transform, testdata_Factor_Mode_Transform)

write.table(k_means_results$k_means_training, sep="\t", na = "" ,  file = "upselling_Step_2B_train_Factor_Mode_Transform_k_means.csv", row.names = FALSE)
write.table(k_means_results$k_means_test, sep="\t", na = "",  file = "upselling_Step_2B_test_Factor_Mode_Transform_k_means.csv", row.names = FALSE)

# stops the cluster

stopCluster(cl)


################ save output 

save.image(file = "Upselling_Step_2B_factor_k_means_bin.RData")
