library(tidyverse)
library(stats)
#Read in the dataset
crime_against_person_statewise_data = read_csv("Crimes_Against_Persons_Offenses_Offense_Category_by_State_2020.csv")
crime_against_person_locationwise_data = read_csv("Crimes_Against_Persons_Offenses_Offense_Category_by_Location_2020.csv")
crime_against_society_statewise_data = read_csv("Crimes_Against_Society_Offenses_Offense_Category_by_State_2020.csv")
crime_against_society_locationwise_data = read_csv("Crimes_Against_Society_Offenses_Offense_Category_by_Location_2020.csv")

####DATA CLEANIND FOR CRIME AGAINST PERSON STATEWISE####
#Column names
ColNames = colnames(crime_against_person_statewise_data); ColNames
#Number of rows and columns before cleaning
NumColumns = ncol(crime_against_person_statewise_data ); NumColumns
NumRows = nrow(crime_against_person_statewise_data); NumRows
##rename the columns
colnames(crime_against_person_statewise_data)[which(names(crime_against_person_statewise_data) == "Total\nOffenses")] = "Total_Offenses"
colnames(crime_against_person_statewise_data)[which(names(crime_against_person_statewise_data) == "Number\nof Participating\nAgencies")] = "Number_of_Participating_Agencies"
colnames(crime_against_person_statewise_data)[which(names(crime_against_person_statewise_data) == "Population\nCovered")] = "Population_Covered"
colnames(crime_against_person_statewise_data)[which(names(crime_against_person_statewise_data) == "Homicide\nOffenses")] = "Homicide_Offenses"
colnames(crime_against_person_statewise_data)[which(names(crime_against_person_statewise_data) == "Human\nTrafficking")] = "Human_Trafficking"
colnames(crime_against_person_statewise_data)[which(names(crime_against_person_statewise_data) == "Kidnapping/\nAbduction")] = "Kidnapping_or_Abduction"
colnames(crime_against_person_statewise_data)[which(names(crime_against_person_statewise_data) == "Sex\nOffenses")] = "Sex_Offenses"
colnames(crime_against_person_statewise_data)[which(names(crime_against_person_statewise_data) == "Assault\nOffenses")] = "Assault_Offenses"
##check if there are Nan values
anyNA(crime_against_person_statewise_data)
#find any dupliactes in  the data
sum_of_duplicates=sum(duplicated(crime_against_person_statewise_data));sum_of_duplicates
# type of data
str(crime_against_person_statewise_data)
#summary statistics of the clean data
lapply(crime_against_person_statewise_data,summary) 

####DATA CLEANIND FOR CRIME AGAINST PERSON LOCATIONWISE####
#Column names
ColNames = colnames(crime_against_person_locationwise_data); ColNames
#Number of rows and columns before cleaning
NumColumns = ncol(crime_against_person_locationwise_data ); NumColumns
NumRows = nrow(crime_against_person_locationwise_data ); NumRows
##rename the columns 
colnames(crime_against_person_locationwise_data)[which(names(crime_against_person_locationwise_data) == "Total\nOffenses")] = "Total_Offenses"
colnames(crime_against_person_locationwise_data)[which(names(crime_against_person_locationwise_data) == "Assault\nOffenses")] = "Assault_Offenses"
colnames(crime_against_person_locationwise_data)[which(names(crime_against_person_locationwise_data) == "Homicide\nOffenses")] = "Homicide_Offenses"
colnames(crime_against_person_locationwise_data)[which(names(crime_against_person_locationwise_data) == "Human\nTrafficking")] = "Human_Trafficking"
colnames(crime_against_person_locationwise_data)[which(names(crime_against_person_locationwise_data) == "Kidnapping/\nAbduction")] = "Kidnapping_or_Abduction"
colnames(crime_against_person_locationwise_data)[which(names(crime_against_person_locationwise_data) == "Sex\nOffenses")] = "Sex_Offenses"

##check if there are Nan values
anyNA(crime_against_person_locationwise_data)
#find any dupliactes in  the data
sum_of_duplicates=sum(duplicated(crime_against_person_locationwise_data));sum_of_duplicates
# type of data
str(crime_against_person_locationwise_data)
#summary statistics of the clean data
lapply(crime_against_person_locationwise_data,summary) 

####DATA CLEANIND FOR CRIME AGAINST SOCIETY STATEWISE####
#Column names
ColNames = colnames(crime_against_society_statewise_data); ColNames
#Number of rows and columns before cleaning
NumColumns = ncol(crime_against_society_statewise_data ); NumColumns
NumRows = nrow(crime_against_society_statewise_data ); NumRows
##rename the columns
colnames(crime_against_society_statewise_data)[which(names(crime_against_society_statewise_data) =="Number of\nParticipating\nAgencies")] = "Number_of_Participating_Agencies"
colnames(crime_against_society_statewise_data)[which(names(crime_against_society_statewise_data) == "Population\nCovered")] = "Population_Covered"
colnames(crime_against_society_statewise_data)[which(names(crime_against_society_statewise_data) == "Total\nOffenses")] = "Total_Offenses"
colnames(crime_against_society_statewise_data)[which(names(crime_against_society_statewise_data) == "Animal\nCruelty")] = "Animal_Cruelty"
colnames(crime_against_society_statewise_data)[which(names(crime_against_society_statewise_data) == "Drug/\nNarcotic\nOffenses")] = "Drug_or_Narcotic_Offenses"
colnames(crime_against_society_statewise_data)[which(names(crime_against_society_statewise_data) == "Gambling\nOffenses")] = "Gambling_Offenses"
colnames(crime_against_society_statewise_data)[which(names(crime_against_society_statewise_data) == "Pornography/\nObscene\nMaterial")] = "Pornography_or_Obscene_Material"
colnames(crime_against_society_statewise_data)[which(names(crime_against_society_statewise_data) == "Prostitution\nOffenses")] = "Prostitution_Offenses"
colnames(crime_against_society_statewise_data)[which(names(crime_against_society_statewise_data) == "Weapon\nLaw\nViolations")] = "Weapon_Law_Violations"
##check if there are Nan values
anyNA(crime_against_society_statewise_data)
#find any dupliactes in  the data
sum_of_duplicates=sum(duplicated(crime_against_society_statewise_data));sum_of_duplicates
# type of data
str(crime_against_society_statewise_data)
#summary statistics of the clean data
lapply(crime_against_society_statewise_data,summary) 

####DATA CLEANIND FOR CRIME AGAINST SOCIETY LOCATIONWISE####

#Column names
ColNames = colnames(crime_against_society_locationwise_data); ColNames
#Number of rows and columns before cleaning
NumColumns = ncol(crime_against_society_locationwise_data ); NumColumns
NumRows = nrow(crime_against_society_locationwise_data ); NumRows
##rename the columns
colnames(crime_against_society_locationwise_data)[which(names(crime_against_society_locationwise_data) == "Total\nOffenses")] = "Total_Offenses"
colnames(crime_against_society_locationwise_data)[which(names(crime_against_society_locationwise_data) == "Animal\nCruelty")] = "Animal_Cruelty"
colnames(crime_against_society_locationwise_data)[which(names(crime_against_society_locationwise_data) == "Drug/\nNarcotic\nOffenses")] = "Drug_or_Narcotic_Offenses"
colnames(crime_against_society_locationwise_data)[which(names(crime_against_society_locationwise_data) == "Gambling\nOffenses")] = "Gambling_Offenses"
colnames(crime_against_society_locationwise_data)[which(names(crime_against_society_locationwise_data) == "Pornography/\nObscene\nMaterial")] = "Pornography_or_Obscene_Material"
colnames(crime_against_society_locationwise_data)[which(names(crime_against_society_locationwise_data) == "Prostitution\nOffenses")] = "Prostitution_Offenses"
colnames(crime_against_society_locationwise_data)[which(names(crime_against_society_locationwise_data) == "Weapon\nLaw\nViolations")] = "Weapon_Law_Violations"
##check if there are Nan values
anyNA(crime_against_society_locationwise_data)
#find any dupliactes in  the data
sum_of_duplicates=sum(duplicated(crime_against_society_locationwise_data));sum_of_duplicates
# type of data
str(crime_against_society_locationwise_data)
#summary statistics of the clean data
lapply(crime_against_society_locationwise_data,summary)



#Visualizations
library(ggplot2)
library(stringr)


##State-wise crime against person 

total_crime_statewise = data.frame(crime_against_person_statewise_data$State, crime_against_person_statewise_data$Total_Offenses)
filter_total_crime_statewise=total_crime_statewise[2: 48 , ]; filter_total_crime_statewise
filter_total_crime_statewise %>%
  mutate(crime_against_person_statewise_data.State = str_remove(crime_against_person_statewise_data.State, "-.*$")) %>%
  mutate(crime_against_person_statewise_data.State = str_remove(crime_against_person_statewise_data.State, ",.*$")) %>% 
  ggplot(aes(y = reorder(crime_against_person_statewise_data.State, crime_against_person_statewise_data.Total_Offenses), x = crime_against_person_statewise_data.Total_Offenses)) +
  geom_col(stat="identity",color="black", fill="#E57373")+
  ggtitle("Crime against Person in US by State")+
  xlab("Crime rate in thousands")+
  ylab("States")+
  theme_bw()

##Crime against person location-wise

total_crime_locationwise = data.frame(crime_against_person_locationwise_data$Location, crime_against_person_locationwise_data$Total_Offenses)
filter_total_crime_locationwise=total_crime_locationwise[2: 47 , ]; filter_total_crime_locationwise
#options(repr.plot.width = 50, repr.plot.height = 100)
filter_total_crime_locationwise %>%
  mutate(crime_against_person_locationwise_data.Location = str_remove(crime_against_person_locationwise_data.Location, "-.*$")) %>%
  mutate(crime_against_person_locationwise_data.Location = str_remove(crime_against_person_locationwise_data.Location, ",.*$")) %>% 
  ggplot(aes(y = reorder(crime_against_person_locationwise_data.Location, crime_against_person_locationwise_data.Total_Offenses), x = crime_against_person_locationwise_data.Total_Offenses)) +
  geom_col(stat="identity", color="black", fill="#CE93C8")+
  ggtitle("Crime against Person in US by Location")+
  xlab("Crime rate in thousands")+
  ylab("Location")+
  theme_bw()

##Crime against society state-wise
total_crime_society_statewise = data.frame(crime_against_society_statewise_data$State, crime_against_society_statewise_data$Total_Offenses)
filter_total_crime_society_statewise=total_crime_society_statewise[2: 48 , ]; filter_total_crime_society_statewise
filter_total_crime_society_statewise %>%
  mutate(crime_against_society_statewise_data.State = str_remove(crime_against_society_statewise_data.State, "-.*$")) %>%
  mutate(crime_against_society_statewise_data.State = str_remove(crime_against_society_statewise_data.State, ",.*$")) %>% 
  ggplot(aes(y = reorder(crime_against_society_statewise_data.State, crime_against_society_statewise_data.Total_Offenses), x = crime_against_society_statewise_data.Total_Offenses)) +
  geom_col(stat="identity",color="black", fill="#9FA8DA")+
  ggtitle("Crime against Society in US by State")+
  xlab("Crime rate in thousands")+
  ylab("States")+
  theme_bw()

##Crime against society location-wise

total_crime_society_locationwise = data.frame(crime_against_society_locationwise_data$Location, crime_against_society_locationwise_data$Total_Offenses)
filter_total_crime_society_locationwise=total_crime_society_locationwise[2: 47 , ]; filter_total_crime_society_locationwise
#options(repr.plot.width = 50, repr.plot.height = 100)
filter_total_crime_society_locationwise %>%
  mutate(crime_against_society_locationwise_data.Location = str_remove(crime_against_society_locationwise_data.Location, "-.*$")) %>%
  mutate(crime_against_society_locationwise_data.Location = str_remove(crime_against_society_locationwise_data.Location, ",.*$")) %>% 
  ggplot(aes(y = reorder(crime_against_society_locationwise_data.Location, crime_against_society_locationwise_data.Total_Offenses), x = crime_against_society_locationwise_data.Total_Offenses)) +
  geom_col(stat="identity",color="black", fill="#BBDEFB")+
  ggtitle("Crime against Society in US by Location")+
  xlab("Crime rate in thousands")+
  ylab("Location")+
  theme_bw()

##Total Count of Crime by person 
crime_person_by_catergory_data= data.frame(crime_against_person_statewise_data$Assault_Offenses, crime_against_person_statewise_data$Homicide_Offenses,crime_against_person_statewise_data$Human_Trafficking,crime_against_person_statewise_data$Kidnapping_or_Abduction, crime_against_person_statewise_data$Sex_Offenses)
filter_crime_person_by_catergory=crime_person_by_catergory_data[1 ,1:5 ]
filter_crime_person_by_catergory = as.data.frame(t(filter_crime_person_by_catergory))
str(filter_crime_person_by_catergory)
colnames(filter_crime_person_by_catergory) = c("Values")
Category = c("Assault_Offenses","Homicide_Offenses", "Human_Trafficking", "Kidnapping_or_Abduction", "Sex_Offenses")
Values = filter_crime_person_by_catergory$Values
crime_person_by_catergory  = data.frame(Category,Values); crime_person_by_catergory



##Total Count of Crime by society 
crime_society_by_catergory_data = data.frame(crime_against_society_statewise_data$Animal_Cruelty, crime_against_society_statewise_data$Drug_or_Narcotic_Offenses,crime_against_society_statewise_data$Gambling_Offenses,crime_against_society_statewise_data$Pornography_or_Obscene_Material, crime_against_society_statewise_data$Prostitution_Offenses, crime_against_society_statewise_data$Weapon_Law_Violations)
filter_crime_society_by_catergory=crime_society_by_catergory_data[1 ,1:6 ]
filter_crime_society_by_catergory = as.data.frame(t(filter_crime_society_by_catergory))
str(filter_crime_society_by_catergory)
colnames(filter_crime_society_by_catergory) = c("Values")
Category = c("Animal_Cruelty","Drug_or_Narcotic_Offenses", "Gambling_Offenses", "Pornography_or_Obscene_Material", "Prostitution_Offenses","Weapon_Law_Violations")
Values = filter_crime_society_by_catergory$Values

crime_society_by_catergory  = data.frame(Category,Values); crime_society_by_catergory



##line graph for Count of Crime by person 
a=ggplot(crime_person_by_catergory,aes(x=Category, y= Values, group = 1))+
  geom_line(size = 1.5, 
            color = "lightgrey")+
  geom_point(size = 3, 
             alpha = 1/4, color= 62)+
  scale_x_discrete(labels=c("Assault Offenses","Homicide Offenses", "Human Trafficking", "Kidnapping or Abduction", "Sex Offenses"))+
  ggtitle("Crime by Person")


##line graph  for Count of Crime by Society
b=ggplot(crime_society_by_catergory,aes(x=Category, y= Values, group = 1))+
  geom_line(size = 1.5, 
            color = "beige")+
  geom_point(size = 3, 
             alpha = 1/4, color= 140)+
  scale_x_discrete(labels=c("Animal Cruelty","Drug or Narcotic", "Gambling", "Pornography or Obscene","Prostitution","Weapon Law Violations"))+
  ggtitle("Crime by society")

library("patchwork")
all = (a+b) +
  plot_annotation(title="Crime Catergory Distribution ")&theme(plot.title = element_text(size =20, hjust = 0.5))
all

##Top 5 states v/s offenses against person

top_5_states_person=crime_against_person_statewise_data[order(crime_against_person_statewise_data$Total_Offenses, decreasing = TRUE),]
top_5_states_person=top_5_states_person[2:6,]
drop = c("Number_of_Participating_Agencies","Population_Covered","Total_Offenses")
top_5_states_person = top_5_states_person[,!(names(top_5_states_person) %in% drop)]
top_5_states_person
a=ggplot(top_5_states_person, 
         aes(x=top_5_states_person$State,
             y= top_5_states_person$Assault_Offenses))+
  geom_bar(stat="identity", color="black", fill="#E6EE9C")+
  scale_x_discrete(labels=c("MI", "NC", "Ohio","TN","Texas"))+
  ggtitle("Assault Offenses")+xlab(NULL)+ylab(NULL)
b=ggplot(top_5_states_person, 
         aes(x=top_5_states_person$State,
             y= top_5_states_person$Homicide_Offenses))+
  geom_bar(stat="identity",color="black",fill="#F0F4C3")+
  scale_x_discrete(labels=c("MI", "NC", "Ohio","TN","Texas"))+
  ggtitle("Homicide Offenses")+xlab(NULL)+ylab(NULL)
c= ggplot(top_5_states_person, 
          aes(x=top_5_states_person$State,
              y= top_5_states_person$Human_Trafficking))+
  geom_bar(stat="identity",color="black", fill="#DCEDC8")+
  scale_x_discrete(labels=c("MI", "NC", "Ohio","TN","Texas"))+
  ggtitle("Human Trafficking ")+xlab(NULL)+ylab(NULL)
d=ggplot(top_5_states_person, 
         aes(x=top_5_states_person$State,
             y= top_5_states_person$Kidnapping_or_Abduction))+
  geom_bar(stat="identity",color="black", fill="#C8E6C9")+
  scale_x_discrete(labels=c("MI", "NC", "Ohio","TN","Texas"))+
  ggtitle("Kidnapping or Abduction Offenses ")+xlab(NULL)+ylab(NULL)
e=ggplot(top_5_states_person, 
         aes(x=top_5_states_person$State,
             y= top_5_states_person$Sex_Offenses))+
  geom_bar(stat="identity", color="black", fill="#B2DFDB")+
  scale_x_discrete(labels=c("MI", "NC", "Ohio","TN","Texas"))+
  ggtitle("Sex Offenses")+xlab(NULL)+ylab(NULL)

all = (a+b+c)/(d+e) +
  plot_annotation(title="Distribution of Crime by Person for Top 5 States")&theme(plot.title = element_text(hjust = 0.5))
all

##Top 5 states v/s offenses against society 

top_5_states_society=crime_against_society_statewise_data[order(crime_against_society_statewise_data$Total_Offenses, decreasing = TRUE),]
top_5_states_society=top_5_states_society[2:6,]
drop = c("Number_of_Participating_Agencies","Population_Covered","Total_Offenses")
top_5_states_society = top_5_states_society[,!(names(top_5_states_society) %in% drop)]
top_5_states_society
a=ggplot(top_5_states_society, 
         aes(x=top_5_states_society$State,
             y= top_5_states_society$Animal_Cruelty))+
  geom_bar(stat="identity",color="black", fill="#BBDEFB")+
  ggtitle("Animal Cruelty")+xlab(NULL)+ylab(NULL)
b=ggplot(top_5_states_society, 
         aes(x=top_5_states_society$State,
             y= top_5_states_society$Drug_or_Narcotic_Offenses))+
  geom_bar(stat="identity",color="black", fill="#C5CAE9")+
  scale_x_discrete(labels=c("NC","Ohio","TN","Texas","VA"))+
  ggtitle("Drug or Narcotic Offenses")+xlab(NULL)+ylab(NULL)
c=ggplot(top_5_states_society, 
         aes(x=top_5_states_society$State,
             y= top_5_states_society$Gambling_Offenses))+
  geom_bar(stat="identity", color="black", fill="#D1C4E9")+
  scale_x_discrete(labels=c("NC","Ohio","TN","Texas","VA"))+
  ggtitle("Gambling Offenses")+xlab(NULL)+ylab(NULL)
d= ggplot(top_5_states_society, 
          aes(x=top_5_states_society$State,
              y= top_5_states_society$Pornography_or_Obscene_Material))+
  geom_bar(stat="identity",color="black", fill="#E1BEE7")+
  scale_x_discrete(labels=c("NC","Ohio","TN","Texas","VA"))+
  ggtitle("Pornography or Obscene")+xlab(NULL)+ylab(NULL)
e=ggplot(top_5_states_society, 
         aes(x=top_5_states_society$State,
             y= top_5_states_society$Prostitution_Offenses))+
  geom_bar(stat="identity",color="black", fill="#F8BBD0")+
  scale_x_discrete(labels=c("NC","Ohio","TN","Texas","VA"))+
  ggtitle("Prostitution Offenses ")+xlab(NULL)+ylab(NULL)
f=ggplot(top_5_states_society, 
         aes(x=top_5_states_society$State,
             y= top_5_states_society$Weapon_Law_Violations))+
  geom_bar(stat="identity",color="black", fill="#FFCDD2")+
  scale_x_discrete(labels=c("NC","Ohio","TN","Texas","VA"))+
  ggtitle("Weapon Law Violations")+xlab(NULL)+ylab(NULL)

all = (a+b+c)/(d+e+f) +
  plot_annotation(title="Distribution of Crime by Society for Top 5 States")&theme(plot.title = element_text(hjust = 0.5))
all

##Crime against Society annd person v/s the population covered
Population_covered = crime_against_person_statewise_data$Population_Covered
State = crime_against_person_statewise_data$State
person = crime_against_person_statewise_data$Total_Offenses
society = crime_against_society_statewise_data$Total_Offenses 
data = data.frame( State, Population_covered,person, society)
data$total_offense =rowSums(data[,c(3,4)])
data= data[2:48,];data

ggplot(data, aes(x=Population_covered,y=total_offense))+
  geom_point(alpha = 1/4, color = 66)+
  ggtitle("Number of Offenses v/s Population")+
  theme(plot.title = element_text(size = 20, hjust = 0.5))+
  labs(x="Population", y="Number of Offenses")+
  theme(legend.position=c(0,1),legend.justification=c(0,1))

##export data to csv
write_csv(crime_against_person_statewise_data,"../pages/cleaned_crime_against_person_statewise_data_r.csv")
write_csv(crime_against_person_locationwise_data,"../pages/cleaned_crime_against_person_locationwise_data_r.csv")
write_csv(crime_against_society_statewise_data,"../pages/cleaned_crime_against_society_statewise_data_r.csv")
write_csv(crime_against_society_locationwise_data,"../pages/cleaned_crime_against_society_locationwisewise_data_r.csv")
