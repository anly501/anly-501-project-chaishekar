# %% [markdown]
# **DECISION TREE FOR LABELLED RECORD DATA**

# %% [markdown]
# **IMPORT LIBRARIES**

# %%
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn import tree
from IPython.display import Image
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# **ABOUT THE DATA**
# 
# The record data that Decision Tree used relates to the categories of violent crime that include homicide (including murder and non-negligent manslaughter), robbery, and serious assault. Violent crime also includes rape. Offenses classified as violent involve the actual use of, or the threat to use, physical force.
# 
# Only offenses for which a report has been filed are included in the data displayed on the Crime Data Explorer; it is not a comprehensive record of all crimes. Understanding the multiple factors that contribute to criminal behavior and the reporting of crimes in a community is essential before attempting to analyze the data. In the absence of these considerations, the data that are currently available may be deceptive. The size and density of the population, the economy, the unemployment rate, the policies regarding prosecution, the judiciary, and corrections, the administrative and investigative focus of law enforcement, how people feel about crime and the police, and the actual strength of the police force are all important factors to consider.
# 
# The dataset consists of three categories of information regarding victims of violent crimes: their age, gender, and race, together with the various types of crimes that they were victims of.

# %%
# Read cleaned and balanced data
age_data = pd.read_csv('dt_agedata.csv')
race_data = pd.read_csv("svm_racedata.csv")
gender_data = pd.read_csv("dt_genderdata.csv")


# %% [markdown]
# *VICTIMS AGE DATA*

# %%
#HEAD OF DATA
age_data.head()

# %% [markdown]
# *VICTIMS GENDER DATA*

# %%
#HEAD OF DATA
gender_data.head()

# %% [markdown]
# *VICTIMS RACE DATA*

# %%
#HEAD OF DATA
race_data.head()

# %% [markdown]
# **DATA CLEANING**
# 
# For the above datasets, columns that aren't necessary are deleted, and it is determined whether or not the dataset contains any values that are not applicable or duplicates.

# %%
#CLEAN THE DATA
#AGE DATA
#REMOVE THE FIRST COLUMN 
age_data = age_data.drop(columns=['Unnamed: 0', 'Offense'])
#CHECK FOR MISSING VALUES
age_data.isnull().sum()
#GENDER DATA
#REMOVE THE FIRST COLUMN 
gender_data = gender_data.drop(columns=['Unnamed: 0','Offense'])
#CHECK FOR MISSING VALUES
gender_data.isnull().sum()
#RACE DATA
#REMOVE THE FIRST COLUMN
race_data = race_data.drop(columns=['Unnamed: 0'])
#CHECK FOR MISSING VALUES
race_data.isnull().sum()

# %% [markdown]
# *DATA SUMMARY*

# %%
#SUMMARIZE THE AGE DATA

age_data_describe = age_data.describe().loc[['min','mean','max']]
age_data_dtype = age_data.dtypes
age_data_describe = age_data_describe.append(age_data_dtype,ignore_index=True)
age_data_describe = age_data_describe.rename(index={0:'min',1:'mean',2:'max',3:'dtype'})
age_data_describe = age_data_describe.transpose()
print(age_data_describe)

# %%
#SUMMARIZE THE DATA GENDER DATA 
df_describe = gender_data.describe().loc[['min','mean','max']]
df_dtype = gender_data.dtypes
df_describe = df_describe.append(df_dtype,ignore_index=True)
df_describe = df_describe.rename(index={0:'min',1:'mean',2:'max',3:'dtype'})
df_describe = df_describe.transpose()
print(df_describe)

# %%
#SUMMARIZE THE DATA RACE DATA  
df_describe = race_data.describe().loc[['min','mean','max']]
df_dtype = race_data.dtypes
df_describe = df_describe.append(df_dtype,ignore_index=True)
df_describe = df_describe.rename(index={0:'min',1:'mean',2:'max',3:'dtype'})
df_describe = df_describe.transpose()
print(df_describe)

# %% [markdown]
# **DECISION TREE**
# 
# A decision tree is an aid to decision making that employs a tree-like model of decisions and the potential implications, such as the outcomes of random events, the costs and benefits of resources, and the overall value of the decision. If your method consists only of if/then statements, this is one approach to present it.
# 
# As a prominent tool in machine learning, decision trees are also widely used in the field of operations research, particularly in the field of decision analysis, to determine which course of action is most likely to result in the desired outcome.
# 
# Each node inside a decision tree represents a "test" on an attribute (such as whether a coin is headed up or down), each branch reflects the result of that test, and each leaf node represents a class label (decision taken after computing all attributes). The branches stand for different kinds of categorization schemes.

# %% [markdown]
# **Split the dataset into training and testing sets**
# 
# The data is separated into training data and testing data, and each of the three datasets contains unique information. Before continuing, the number of samples that will be used for each sample will be calculated.

# %% [markdown]
# **HYPER-PARAMETERS TUNING**
# 
# The features of the model known as parameters are those that are learned by the model from the data. On the other hand, hyperparameters are arguments that are accepted by a model-making function. These hyperparameters can be adjusted to reduce overfitting, which ultimately results in a model that is more generalizable. The method of hyperparameter tuning, which involves calibrating our model by determining which hyperparameters should be used to extend our model, has been given its own name.

# %% [markdown]
# **FOR VICTIM AGE DATA**

# %% [markdown]
# *Splitting Data into Training and Testing Data in Sklearn*

# %%
#THE LOAD BALANCE AND COUNT THE NUMBER OF SAMPLES FOR EACH CATEGORY
from locale import normalize

Crime_Type_Grave_count = age_data['Crime_Type'].value_counts()['Grave']
Crime_Type_NotGrave_count = age_data['Crime_Type'].value_counts()['NotGrave']
Crime_Type_Grave_norm = age_data['Crime_Type'].value_counts(normalize = True)['Grave']
Crime_Type_NotGrave_norm = age_data['Crime_Type'].value_counts(normalize = True)['NotGrave']

print("Number of points with category Grave: {0:2d} {1:}".format(Crime_Type_Grave_count, Crime_Type_Grave_norm))
print("Number of points with category Not Grave: {0:2d} {1:}".format(Crime_Type_NotGrave_count, Crime_Type_NotGrave_norm))

# %%
#MAKE DATA-FRAMES (or numpy arrays) (X,Y) WHERE Y="category" COLUMN and X="everything else"
X = age_data.drop(columns = ['Crime_Type'])
Y = age_data['Crime_Type']
#PARTITION THE DATASET INTO TRAINING AND TEST SETS
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.62, random_state=2)

#CONSISTENCY CHECK
print(type(x_train))
print(x_train.shape)
print(type(y_train))
print(y_train.shape)
print(type(x_test))
print(x_test.shape)
print(type(y_test))
print(y_test.shape)

# %% [markdown]
# *DECISION TREE MODEL FOR VICTIM AGE DATA*

# %%
#set seed
np.random.seed(2)
# TRAIN A SKLEARN DECISION TREE MODEL ON x_train,y_train 
from sklearn import tree
model = tree.DecisionTreeClassifier()
model = model.fit(x_train, y_train)

#MAKE PREDICTIONS FOR THE TRAINING AND TEST SET 
yp_train = model.predict(x_train)
yp_test = model.predict(x_test)

# %%
#CONFUSION MATRIX 
from sklearn.metrics import confusion_matrix


def confusion_plot(y_data,y_pred):
    cm = confusion_matrix(y_data, y_pred)
    print('ACCURACY: {:.2f}'.format(accuracy_score(y_data, y_pred)))
    print('NEGATIVE RECALL (Y=0): {:.2f}'.format(recall_score(y_data, y_pred, pos_label='Grave')))
    print('NEGATIVE PRECISION (Y=0): {:.2f}'.format(precision_score(y_data, y_pred, pos_label='Grave')))
    print('POSITIVE RECALL (Y=1): {:.2f}'.format(recall_score(y_data, y_pred, pos_label='Grave')))
    print('POSITIVE PRECISION (Y=1): {:.2f}'.format(precision_score(y_data, y_pred, pos_label='NotGrave')))
    print(cm)
    plt.figure(figsize=(10,10))
    sns.heatmap(cm, annot=True, fmt="d", )
    plt.title('Confusion matrix')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()

#TEST SET CONFUSION MATRIX
print("------TEST------")
confusion_plot(y_test,yp_test)

# %% [markdown]
# *DECISION TREE VISUALIZATION*

# %%
# VISUALIZE THE DECISION TREE 
def plot_tree(model,X,Y):
    plt.figure(figsize=(10,10))
    tree.plot_tree(model, feature_names=X.columns, class_names=Y.name, filled=True)
    plt.show()

plot_tree(model,X,Y)

# %% [markdown]
# *HYPER-PARAMETERS TUNING*

# %%
#set seed
np.random.seed(6)
#HYPER-PARAMETERS VALUES
test_results=[]
train_results=[]

for num_layer in range(1,17):
    model = tree.DecisionTreeClassifier(max_depth=num_layer)
    model = model.fit(x_train, y_train)

    yp_train=model.predict(x_train)
    yp_test=model.predict(x_test)

    # print(y_pred.shape)
    test_results.append([num_layer,accuracy_score(y_test, yp_test),recall_score(y_test, yp_test,pos_label='Grave'),recall_score(y_test, yp_test,pos_label='NotGrave')])
    train_results.append([num_layer,accuracy_score(y_train, yp_train),recall_score(y_train, yp_train,pos_label='Grave'),recall_score(y_train, yp_train,pos_label='NotGrave')])



# %%
#### TRAIN A SKLEARN DECISION TREE MODEL ON x_train,y_train 
from sklearn import tree
model = tree.DecisionTreeClassifier(max_depth=18)
model = model.fit(x_train, y_train)

yp_train=model.predict(x_train)
yp_test=model.predict(x_test)

# THE MODEL ON THE TEST SET
print("------TEST------")
confusion_plot(y_test,yp_test)
plot_tree(model,X,Y)

# %% [markdown]
# *INFERENCE FOR VICTIM AGE DECISION MODEL:*
# - The split ratio of teh model is 62% of training data and rest as testinf data, it is categoried with crime type variable(grave and not grave).
# - The accurary of the model before hyper-parametric tuning is 73%.
# - The accuracy of the model after hyper parametric tuning has slightly increased to 82%.
# - The model is not underfitting as accuracy is greater than 50%.
# - The decision tree visulization is about the Age over 65 and Unknown Age of the data, which classifies into different samples. As the sample of the data is very small and the data category is less, the decision tree doesn't have huge classification if the data.
# - As the max_depth increases, the accuracy increases and the optimal tree becomes better.
# 
# NOTE: Since the dataset is small and the accuracy of the model is high, the graph for hyperparametric isn't required as the graph shows similar results for training and testing data.

# %% [markdown]
# **FOR VICTIM GENDER DATA**

# %% [markdown]
# *Splitting Data into Training and Testing Data in Sklearn*

# %%
#THE LOAD BALANCE AND COUNT THE NUMBER OF SAMPLES FOR EACH CATEGORY
from locale import normalize

category_male_count = gender_data['category'].value_counts()['Male']
category_female_count = gender_data['category'].value_counts()['Female']
category_male_norm = gender_data['category'].value_counts(normalize = True)['Male']
category_female_norm = gender_data['category'].value_counts(normalize = True)['Female']

print("Number of points with category white: {0:2d} {1:}".format(category_male_count, category_male_norm))
print("Number of points with category black: {0:2d} {1:}".format(category_female_count, category_female_norm))

# %%
#MAKE DATA-FRAMES (or numpy arrays) (X,Y) WHERE Y="category" COLUMN and X="everything else"
X = gender_data.drop(columns = ['category'])
Y = gender_data['category']
#PARTITION THE DATASET INTO TRAINING AND TEST SETS
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=.52, random_state=2)
#CONSISTENCY CHECK
print(type(x_train))
print(x_train.shape)
print(type(y_train))
print(y_train.shape)
print(type(x_test))
print(x_test.shape)
print(type(y_test))
print(y_test.shape)

# %% [markdown]
# *DECISION TREE MODEL FOR VICTIM GENDER DATA*

# %%
#set seed
np.random.seed(2)
# TRAIN A SKLEARN DECISION TREE MODEL ON x_train,y_train 
from sklearn import tree
model = tree.DecisionTreeClassifier()
model = model.fit(x_train, y_train)
#MAKE PREDICTIONS FOR THE TRAINING AND TEST SET 
yp_train = model.predict(x_train)
yp_test = model.predict(x_test)

# %%
#CONFUSION MATRIX 
from sklearn.metrics import confusion_matrix


def confusion_plot(y_data,y_pred):
    cm = confusion_matrix(y_data, y_pred)
    print('ACCURACY: {:.2f}'.format(accuracy_score(y_data, y_pred)))
    print('NEGATIVE RECALL (Y=0): {:.2f}'.format(recall_score(y_data, y_pred, pos_label='Male')))
    print('NEGATIVE PRECISION (Y=0): {:.2f}'.format(precision_score(y_data, y_pred, pos_label='Male')))
    print('POSITIVE RECALL (Y=1): {:.2f}'.format(recall_score(y_data, y_pred, pos_label='Male')))
    print('POSITIVE PRECISION (Y=1): {:.2f}'.format(precision_score(y_data, y_pred, pos_label='Female')))
    print(cm)
    plt.figure(figsize=(10,10))
    sns.heatmap(cm, annot=True, fmt="d", )
    plt.title('Confusion matrix')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()

#TEST SET CONFUSION MATRIX
print("------TEST------")
confusion_plot(y_test,yp_test)

# %% [markdown]
# *DECISION TREE VISUALIZATION*

# %%
# VISUALIZE THE DECISION TREE 
def plot_tree(model,X,Y):
    plt.figure(figsize=(10,10))
    tree.plot_tree(model, feature_names=X.columns, class_names=Y.name, filled=True)
    plt.show()

plot_tree(model,X,Y)

# %%
#set seed
np.random.seed(676)
#HYPER-PARAMETERS VALUES
test_results=[]
train_results=[]

for num_layer in range(1,17):
    model = tree.DecisionTreeClassifier(max_depth=18)
    model = model.fit(x_train, y_train)

    yp_train=model.predict(x_train)
    yp_test=model.predict(x_test)

    # print(y_pred.shape)
    test_results.append([num_layer,accuracy_score(y_test, yp_test),recall_score(y_test, yp_test,pos_label='Male'),recall_score(y_test, yp_test,pos_label='Female')])
    train_results.append([num_layer,accuracy_score(y_train, yp_train),recall_score(y_train, yp_train,pos_label='Male'),recall_score(y_train, yp_train,pos_label='Female')])



# %%
#### TRAIN A SKLEARN DECISION TREE MODEL ON x_train,y_train 
from sklearn import tree
model = tree.DecisionTreeClassifier(max_depth=18)
model = model.fit(x_train, y_train)

yp_train=model.predict(x_train)
yp_test=model.predict(x_test)

# THE MODEL ON THE TEST SET
print("------TEST------")
confusion_plot(y_test,yp_test)

plot_tree(model,X,Y)

# %% [markdown]
# *INFERENCE FOR VICTIM GENDER DECISION MODEL:*
# - The split ratio of teh model is 52% of training data and rest as testinf data, it is categoried with category variable(male and female).
# - The accurary of the model before and after hyper-parametric tuning is 88%.
# - The model is not underfitting as accuracy is greater than 50%.
# - As the max_depth increases, the accuracy increases and the optimal tree becomes better.
# - The decision tree visulization is about the Male and Female category of the data, which classifies into different samples.
# 
# NOTE: Since the dataset is small and the accuracy of the model is high, the graph for hyperparametric isn't required as the graph shows similar results for training and testing data.

# %% [markdown]
# **FOR VICTIM RACE DATA**

# %% [markdown]
# *Splitting Data into Training and Testing Data in Sklearn*

# %%
#THE LOAD BALANCE AND COUNT THE NUMBER OF SAMPLES FOR EACH CATEGORY
from locale import normalize

category_white_count = race_data['category'].value_counts()['White']
category_black_count = race_data['category'].value_counts()['Black_or_African_American']
category_white_norm = race_data['category'].value_counts(normalize = True)['White']
category_black_norm = race_data['category'].value_counts(normalize = True)['Black_or_African_American']

print("Number of points with category white: {0:2d} {1:}".format(category_white_count, category_white_norm))
print("Number of points with category black: {0:2d} {1:}".format(category_black_count, category_black_norm))

# %%
#MAKE DATA-FRAMES (or numpy arrays) (X,Y) WHERE Y="category" COLUMN and X="everything else"
X = race_data.drop(columns = ['category'])
Y = race_data['category']
#PARTITION THE DATASET INTO TRAINING AND TEST SETS
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=.52, random_state=2)
#CONSISTENCY CHECK
print(type(x_train))
print(x_train.shape)
print(type(y_train))
print(y_train.shape)
print(type(x_test))
print(x_test.shape)
print(type(y_test))
print(y_test.shape)

# %% [markdown]
# *DECISION TREE MODEL FOR VICTIM RACE DATA*

# %%
#set seed
np.random.seed(2)
# TRAIN A SKLEARN DECISION TREE MODEL ON x_train,y_train 
from sklearn import tree
model = tree.DecisionTreeClassifier()
model = model.fit(x_train, y_train)
#MAKE PREDICTIONS FOR THE TRAINING AND TEST SET 
yp_train = model.predict(x_train)
yp_test = model.predict(x_test)

# %%
#CONFUSION MATRIX 
from sklearn.metrics import confusion_matrix


def confusion_plot(y_data,y_pred):
    cm = confusion_matrix(y_data, y_pred)
    print('ACCURACY: {:.2f}'.format(accuracy_score(y_data, y_pred)))
    print('NEGATIVE RECALL (Y=0): {:.2f}'.format(recall_score(y_data, y_pred, pos_label='White')))
    print('NEGATIVE PRECISION (Y=0): {:.2f}'.format(precision_score(y_data, y_pred, pos_label='White')))
    print('POSITIVE RECALL (Y=1): {:.2f}'.format(recall_score(y_data, y_pred, pos_label='White')))
    print('POSITIVE PRECISION (Y=1): {:.2f}'.format(precision_score(y_data, y_pred, pos_label='Black_or_African_American')))
    print(cm)
    plt.figure(figsize=(10,10))
    sns.heatmap(cm, annot=True, fmt="d", )
    plt.title('Confusion matrix')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()

#TEST SET CONFUSION MATRIX
print("------TEST------")
confusion_plot(y_test,yp_test)

# %%
# VISUALIZE THE DECISION TREE 
def plot_tree(model,X,Y):
    plt.figure(figsize=(10,10))
    tree.plot_tree(model, feature_names=X.columns, class_names=Y.name, filled=True)
    plt.show()

plot_tree(model,X,Y)

# %%
#set seed
np.random.seed(67)
#HYPER-PARAMETERS VALUES
test_results=[]
train_results=[]

for num_layer in range(1,17):
    model = tree.DecisionTreeClassifier(max_depth=12)
    model = model.fit(x_train, y_train)

    yp_train=model.predict(x_train)
    yp_test=model.predict(x_test)

    # print(y_pred.shape)
    test_results.append([num_layer,accuracy_score(y_test, yp_test),recall_score(y_test, yp_test,pos_label='White'),recall_score(y_test, yp_test,pos_label='Black_or_African_American')])
    train_results.append([num_layer,accuracy_score(y_train, yp_train),recall_score(y_train, yp_train,pos_label='White'),recall_score(y_train, yp_train,pos_label='Black_or_African_American')])



# %%
#### TRAIN A SKLEARN DECISION TREE MODEL ON x_train,y_train 
from sklearn import tree
model = tree.DecisionTreeClassifier(max_depth=50)
model = model.fit(x_train, y_train)

yp_train=model.predict(x_train)
yp_test=model.predict(x_test)

# THE MODEL ON THE TEST SET
print("------TEST------")
confusion_plot(y_test,yp_test)

plot_tree(model,X,Y)

# %% [markdown]
# *INFERENCE FOR VICTIM RACE DECISION MODEL:*
# - The split ratio of teh model is 52% of training data and rest as testinf data, it is categoried with race category variable.
# - The accurary of the model before and after hyper-parametric tuning is 89%.
# - The model is not underfitting as accuracy is greater than 50%.
# - As the max_depth increases, the accuracy increases and the optimal tree becomes better.
# - The decision tree visulization is about White and Pacific Islander category of the data, which classifies into different samples.
# 
# NOTE: Since the dataset is small and the accuracy of the model is high, the graph for hyperparametric isn't required as the graph shows similar results for training and testing data.

# %% [markdown]
# **CONCLUSION**
# 
# The purpose of this Decision Tree study was to categorize the age, race, and gender category based on the sorts of crimes committed. After hyper-parametric tuning, the model achieves an impressively high accuracy of 82% in predicting age, 88% in predicting gender, and 88% in predicting race. The tuning graph is not displayed above because the model performs similarly on both the training and testing data, which is limited in size.


