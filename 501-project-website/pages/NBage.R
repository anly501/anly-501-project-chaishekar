# Libraries
library(e1071)
library(caTools)
library(caret)
library(dplyr)
library(Hmisc)
library(tidyverse)
library(ggplot2)
library(caret)
library(caretEnsemble)
library(psych)
library(Amelia)
library(mice)
library(GGally)
library(ROSE)
library(rpart)
library(randomForest)

#read the age data
age_data = read_csv("Victims_Age_by_Offense_Category_2021.csv")

#delete the total row
age_data = age_data[-c(2,8), ]
#delete unwanted column
#drop = c("Total")
#age_data = age_data[,!(names(age_data) %in% drop)]
##check if there are Nan values
anyNA(age_data)
#find any dupliactes in  the data
sum_of_duplicates=sum(duplicated(age_data));sum_of_duplicates
# structure id the dataset
str(age_data)
#summary statistics of the clean data
lapply(age_data,summary) 


# Balancing Data
age_balanced <- ovun.sample(Crime_Type~., data=age_data,
                            N=nrow(age_data), p=0.5,
                            seed=1, method="both")$data

apply(age_balanced, 2, table)

# Exploratory data analysis
describe(age_balanced)

# Splitting test and train

split <- sample.split(age_balanced, SplitRatio = 0.52)
train <- subset(age_balanced, split == "TRUE")
test <- subset(age_balanced, split == "FALSE")

# Setting Seed
set.seed(663)  
# Model Building
model <- naiveBayes(Crime_Type ~ ., data = train)
model

# Prediction
Predict <- predict(model, newdata = test)
plot(Predict, main="Distribution of Predict Age Data", col ="#524B64")
# Confusion Matrix 
cm <- table(test$Crime_Type, Predict)
confusionMatrix(cm)


# Graphs 
(misclass <- table(predict = Predict, truth = test$Crime_Type))
fourfoldplot(misclass,color = c("#80A8B6","#3A849F"),
             main = "Confusion Matrix for Age Distribution")


