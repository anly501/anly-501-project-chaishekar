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
consumerKey = 'LBNeNod2Bd3Y0AdGBJT8Ka4Aa'
consumerSecret = 'MNlkJwxEeqS0MXkpi2wNidrAbiUpf4ueVkBJcvZsBBtvne1A7F'
access_Token = '1568064617036881922-LI9dm4r6ISllYfbKE4m9Vz2OW0MZI5'
access_Secret = 'MCq785e0hNBUkkkSwmQFHglVs6hMMKFh1pylkpCCVwyGa'

requestURL='https://api.twitter.com/oauth/request_token'
accessURL='https://api.twitter.com/oauth/access_token'
authURL='https://api.twitter.com/oauth/authorize'

setup_twitter_oauth(consumerKey,consumerSecret,access_Token,access_Secret)
#search hashtage
Search=twitteR::searchTwitter("#FBI",n=10000, since="2000-01-01", lang="en")
(Search_DF = twListToDF(Search))

(Search_DF$text[1])

write.csv(Search_DF,"R_FBI_API.csv")


File_Name = "R_FBI_API.txt"
#Start the file
MyFile <- file(File_Name)
#Write Tweets to file
cat(unlist(Search_DF), " ", file=MyFile, sep="\n\n\n")
close(MyFile)

