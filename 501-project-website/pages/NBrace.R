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
race_data = read_csv("Victims_Race_by_Offense_Category_2021.csv")
#####cleaning race data######
#Column names
ColNames = colnames(race_data); ColNames
##rename the columns
colnames(race_data)[which(names(race_data) == "Black\nor African\nAmerican")] = "Black_or_African_American"
colnames(race_data)[which(names(race_data) == "American\nIndian or\nAlaska Native")] = "American_Indian_or_Alaska_Native"
colnames(race_data)[which(names(race_data) == "Native\nHawaiian or\nOther Pacific\nIslander")] = "Native_Hawaiian_or_Other_Pacific_Islander"
colnames(race_data)[which(names(race_data) == "Unknown\nRace")] = "Unknown"
#delete the total row
race_data = race_data[-c(2,8), ]
# race_data <- as.matrix(race_data)
#delete unwanted column
drop = c("Total")
race_data = race_data[,!(names(race_data) %in% drop)]
##check if there are Nan values
anyNA(race_data)
#find any dupliactes in  the data
sum_of_duplicates=sum(duplicated(race_data));sum_of_duplicates
# structure id the dataset
str(race_data)
#summary statistics of the clean data
lapply(race_data,summary) 
race_data1 <- race_data[,-1]
race_data1<-as.matrix(race_data1)
category <- rowMaxs(race_data1)
race_data1<-as.data.frame(race_data1)
#race_data1<-cbind(race_data1,category)

race_data1
race_data2 = cbind(race_data1, colnames(race_data1)[apply(race_data1,1,which.max)])

colnames(race_data2) = c("White", "Black_or_African_American"                            
                         ,"American_Indian_or_Alaska_Native", "Asian"                                                
                         , "Native_Hawaiian_or_Other_Pacific_Islander", "Unknown"                                              
                         , "category")

race_data2
# Balancing Data
race_balanced <- ovun.sample(category~., data=race_data2,
                             N=nrow(race_data2), p=0.5,
                             seed=1, method="both")$data

apply(race_balanced, 2, table)

# Exploratory data analysis
describe(race_balanced)

# Splitting test and train
set.seed(11)
indxTrain <- createDataPartition(y = race_balanced$category,p = 0.75,list = FALSE)
train <- race_balanced[indxTrain,]
test <- race_balanced[-indxTrain,] #Check dimensions of the split > pro

# Feature Scaling
x = train[,-7]
y = train$category

# Model Building
set.seed(123)
model = train(x,y,'nb',trControl=trainControl(method='cv',number=10))
model

# Prediction
Predict <- predict(model,newdata = test ) #Get the confusion matrix to see accuracy value and other parameter values > confusionMatrix(Predict, testing$Outcome )
Predict
plot(Predict, main="Distribution of Predict Race Data", col ="#DBEEA6")



# Variable Importance Plot
X <- varImp(model)
plot(X)

# Confusion Matrix 
cm = confusionMatrix(as.factor(test$category), Predict)
print(cm)

# Graphs 
(misclass <- table(predict = Predict, truth = test$category))
misclass = t(misclass)
#rename misclass column name
colnames(misclass)
colnames(misclass) = c("Black or African American", "White")
misclass = t(misclass)
colnames(misclass) = c("Black or African American", "White")
misclass = t(misclass)
#confusion matrix plot
fourfoldplot(misclass, color = c("#E5FFCC","#CCFFFF"),
             main = "Confusion Matrix for Race Distribution")


