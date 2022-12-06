
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
##import clean data
clean_age_data = pd.read_csv('./age_clean_data_r.csv')

##add a column for total number of crimes by age group
clean_age_data.loc[:,'Offense Total'] = clean_age_data.sum(numeric_only=True, axis=1)

##melt data to get age group and offense type in one column
clean_age_data_melt = pd.melt(clean_age_data, id_vars=['Offenses'],var_name="Age group", value_name = 'Offense Count')
clean_age_data_melt = clean_age_data_melt[clean_age_data_melt['Age group'] != 'Offense Total']
clean_age_data_melt = clean_age_data_melt[clean_age_data_melt['Age group'] != 'Offenses']
clean_age_data_melt = clean_age_data_melt[clean_age_data_melt['Age group'] != 'Total']

#line plot of total number of crimes by age group
sns.set_style("whitegrid")
sns.set_context("talk")
sns.set_palette("Set2")
sns.lineplot(x="Age group", y="Offense Count", data=clean_age_data_melt)
plt.title('Total Number of Crimes by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Total Number of Crimes')


plt.show()


# Visulation for Clean data
# which is the top and least 10 commited crimes?
##calculating top and least 10 crimes 
#top 10 offense type 
top_10_offense = clean_age_data.sort_values(by=['Offense Total'], ascending=False).head(10)
top_10_offense
#least 10 offense type
least_10_offense = clean_age_data.sort_values(by=['Offense Total'], ascending=False).tail(10)
least_10_offense


##barplot for top 10 offense type and least 10 offense type
fig, ax = plt.subplots(1,2, figsize=(18, 3))
sns.barplot(x='Offense Total', y='Offenses', data=top_10_offense, palette=sns.color_palette("GnBu_d"), ax=ax[0])
sns.barplot(x='Offense Total', y='Offenses', data=least_10_offense, palette="vlag", ax=ax[1])
ax[0].set_title('Top 10 Offense Type', fontsize=20)
ax[0].set_xlabel('Count in million', fontsize=15)
ax[0].set_ylabel('Offense', fontsize=15)
ax[1].set_title('Least 10 Offense Type', fontsize=20)
ax[1].set_xlabel('Count in thousands', fontsize=15)
ax[1].set_ylabel('Offense', fontsize=15)
fig.set_size_inches(18.5, 10.5)
plt.tight_layout()
plt.show()



##which is the top and least 5 crimes by each age group
##top 5 crimes by each age group

top_5_offense= top_10_offense.head(5)
top_5_offense= top_5_offense.drop(['Offense Total'], axis=1)
##least 5 crimes by each age group
least_5_offense= least_10_offense.tail(5)
least_5_offense= least_5_offense.drop(['Offense Total'], axis=1)
##melting the data for plotting
df = pd.melt(top_5_offense, id_vars="Offenses", var_name="Age Group", value_name="Count")
df1 = pd.melt(least_5_offense, id_vars="Offenses", var_name="Age Group", value_name="Count")
##plotting the data
top_5_offense_graph=sns.factorplot("Age Group", 'Count', col='Offenses', data=df, kind='bar', col_wrap=5,  palette=sns.color_palette("Greys"))
top_5_offense_graph.fig.subplots_adjust(top=0.8)
top_5_offense_graph.fig.suptitle('Top 5 Offenses by Age Group', fontsize=20)

least_5_offense_graph=sns.factorplot("Age Group", 'Count', col='Offenses', data=df1, kind='bar', col_wrap=5,  palette=sns.cubehelix_palette(8))
least_5_offense_graph.fig.subplots_adjust(top=0.8)
least_5_offense_graph.fig.suptitle('Least 5 Offenses by Age Group', fontsize=20)

plt.show()




