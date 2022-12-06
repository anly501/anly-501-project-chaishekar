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
#read the race data
gender_data = read_csv("Victims_Sex_by_Offense_Category_2021.csv")
#######cleaning gender data#########
#Column names
ColNames = colnames(gender_data); ColNames
#rename the column
colnames(gender_data)[which(names(gender_data) == "Unknown\nSex")] = "Unknown"
colnames(gender_data)[which(names(gender_data) == "Offense Category")] = "Offense"#delete the total row
gender_data = gender_data[-c(2,8), ]
#delete unwanted column
drop = c("Total","Unknown")
gender_data = gender_data[,!(names(gender_data) %in% drop)]
##check if there are Nan values
anyNA(gender_data)
#find any dupliactes in  the data
sum_of_duplicates=sum(duplicated(gender_data));sum_of_duplicates
# structure id the dataset
str(gender_data)
#summary statistics of the clean data
lapply(gender_data,summary) 

gender_data_nb = gender_data
gender_data_nb$category = gender_data_nb$Female
gender_data_nb$category[gender_data_nb$Female>gender_data_nb$Male] = "Female"
gender_data_nb$category[gender_data_nb$Female<gender_data_nb$Male] = "Male"
gender_data_nb = gender_data_nb[c(-1),]


# Balancing Data
gender_balanced <- ovun.sample(category~., data=gender_data_nb,
                             N=nrow(gender_data_nb), p=0.5,
                             seed=1, method="both")$data

apply(gender_balanced, 2, table)

# Exploratory data analysis
describe(gender_balanced)

# Splitting test and train
indxTrain <- createDataPartition(y = gender_balanced$category,p = 0.75,list = FALSE)
train <- gender_balanced[indxTrain,]
test <- gender_balanced[-indxTrain,] #Check dimensions of the split > pro

# Feature Scaling
x = train[,-4]
x = x[,-1]
y = train$category

# Model Building
set.seed(123)
model = train(x,y,'nb',trControl=trainControl(method='cv',number=10))
model

# Prediction
Predict <- predict(model,newdata = test ) #Get the confusion matrix to see accuracy value and other parameter values > confusionMatrix(Predict, testing$Outcome )
Predict

plot(Predict, main="Distribution of Predict Gender Data", col ="#45739B")


# Confusion Matrix 
cm = confusionMatrix(as.factor(test$category), Predict)
print(cm)
# Graphs 
(misclass <- table(predict = Predict, truth = test$category))
fourfoldplot(misclass, color = c("#CCE5FF","#FFFFCC"),
             main = "Confusion Matrix for Gender Distribution")

