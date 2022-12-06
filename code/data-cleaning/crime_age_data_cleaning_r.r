library(tidyverse)
library(stats)
#Read in the dataset
age_data = read_csv("ucrAgeGender.csv")

##type of data
str(age_data)

#Column names
ColNames = colnames(age_data); ColNames
#Number of rows and columns before cleaning
NumColumns = ncol(age_data); NumColumns
NumRows = nrow(age_data); NumRows

#remove unwanted columns 
drop = c("All ages", "0 to 17", "18 & older", "10 to 17")
age_data = age_data[,!(names(age_data) %in% drop)]; age_data
#remove NA rows
age_data = age_data[complete.cases(age_data),]; age_data

#remove all offenses
age_data = age_data[-c(1),]

#find any dupliactes in  the data
sum_of_duplicates=sum(duplicated(age_data));sum_of_duplicates

#summary statistics of the clean data
lapply(age_data,summary) 

#export clean data into new csv
write_csv(age_data,"../pages/age_clean_data_r.csv")
