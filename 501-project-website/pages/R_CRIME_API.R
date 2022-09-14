#install.packages("selectr")
#install.packages("rvest")
#install.packages("xml2")
#install.packages("twitteR")
#install.packages("ROAuth")
#install.packages("rtweet")
library("selectr")
library("rvest")
library("xml2")
library(rtweet)
library(twitteR)
library(ROAuth)
library(jsonlite)

#twitter API
consumerKey = 'qYmcNkrPVwugbmvQMaKXn9rMS'
consumerSecret = 'hTVhbc2ZbL2yAvRKWuYhq0bejJsBbxzefTAW3rUOfWjNVO1SSR'
access_Token = '1553029741187215360-zU3nLtr00qJLj7lwGRhxLPVkwxYPID'
access_Secret = '2uSoVz93noHJIp80Zcuu6hlfJxiG45VNPW4ELMXUNG0gA'

requestURL='https://api.twitter.com/oauth/request_token'
accessURL='https://api.twitter.com/oauth/access_token'
authURL='https://api.twitter.com/oauth/authorize'

setup_twitter_oauth(consumerKey,consumerSecret,access_Token,access_Secret)
#search hashtage
Search=twitteR::searchTwitter("#crime",n=10000, since="2000-01-01")
(Search_DF = twListToDF(Search))

(Search_DF$text[1])

write.csv(Search_DF,"R_CRIME_API.csv")


File_Name = "R_CRIME_API.txt"
#Start the file
MyFile <- file(FName)
#Write Tweets to file
cat(unlist(Search_DF), " ", file=MyFile, sep="\n\n\n")
close(MyFile)

