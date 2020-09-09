#!/usr/bin/env python
# coding: utf-8

# # Final Capstone Project

# ## 1. Introduction 

# Car accidents are happending all the time in the world. According to WSDOT's (2017) data, a car accident occurs every 4 minutes and a person dies in a car crash every 20 hours in the state of Washington, U.S.A. To help with reduction of car accident cases, this project trys analysing the determinants of an accident and sheds light on predicting the severity with those factors. 

# ## 2. Data

# The data of car accidents which have occurred within the city of Seattle, Washington from the year 2004 to 2020 was used. This data is regarding the severity of each car accidents along with the time and conditions under which each accident occurred. The model aims to predict the severity of an accident with other information provided. All useful features were extracted and and the missing values were handled at first, followed by a creation of a balanced dataset with equal number of two severity type cases. Lately classfication methods such as KNN, random forest and decision tree were used.

# ### Importing the dataset 

# In[254]:


import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import matplotlib.ticker as ticker
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.utils import resample
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import matplotlib.image as mpimg
from sklearn import tree
from sklearn.tree import export_graphviz
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import matplotlib as mpl


# In[255]:


df = pd.read_csv("Data-Collisions.csv")
df.head()


# In[256]:


df.tail()


# In[257]:


df.rename(columns={'X': 'LONGITUDE', 'Y': 'LATITUDE'}, inplace = True)


# In[258]:


df.shape


# In[259]:


df.dtypes


# ### Feature extraction

# In[260]:


#Drop redundant columns, e.g.,
#     LOCATION: Langitude and Latitude have used.
#     SEVERITYCODE.1: duplicated
#Drop columns with codes: OBJECTID, INCKEY, COLDETKEY, REPORTNO,INTKEY,EXCEPTRSNCODE, SDOT_COLCODE, SDOTCOLNUM
#     ST_COLCODE, ST_COLDESC, SEGLANEKEY, CROSSWALKKEY 
#Drop helpless information: STATUS, EXCEPTRSNDESC, INCDATE , INCDTTM, SDOT_COLDESC, PEDROWNOTGRNT,ST_COLDESC, UNDERINFL--->
#    PEDCYLCOUNT, HITPARKEDCAR, SEVERITYDESC, ADDRTYPE  

df.drop(["LOCATION", "SEVERITYCODE.1", "OBJECTID", "INCKEY", "COLDETKEY", "REPORTNO", "INTKEY", 
          "EXCEPTRSNCODE", "SDOT_COLCODE", "ST_COLCODE", "SEGLANEKEY", "CROSSWALKKEY", "SDOTCOLNUM", 
          "STATUS", "EXCEPTRSNDESC", "INCDATE", "INCDTTM", "SDOT_COLDESC", "PEDROWNOTGRNT", "UNDERINFL", 
        "PEDCYLCOUNT", "HITPARKEDCAR", "ST_COLDESC", "SEVERITYDESC", "ADDRTYPE", "COLLISIONTYPE", "PEDCOUNT"], axis=1, inplace = True)
df.head()


# In[261]:


df.shape


# In[262]:


df.isna().sum()


# ### Dealing with missing values

# In[263]:


#Weather:
df['WEATHER'].value_counts()


# In[264]:


#Weather is very important, we can't just randomly give a value for each, so I opt to drop those rows with missing values
df.dropna(subset = ["WEATHER"], axis = 0, inplace = True)
df.isna().sum()


# In[265]:


encoding_WEATHER = {"WEATHER": 
                            {"Clear": 1,
                             "Unknown": 1,
                             "Other": 1,
                             "Raining": 2,
                             "Overcast": 3,
                             "Snowing": 4,
                             "Fog/Smog/Smoke": 5,
                             "Sleet/Hail/Freezing Rain": 6,
                             "Blowing Sand/Dirt": 7,
                             "Severe Crosswind": 8,
                             "Partly Cloudy": 9}}
df.replace(encoding_WEATHER, inplace=True)
df['WEATHER'].value_counts()


# In[266]:


#Speed:
df['SPEEDING'].value_counts()


# In[267]:


#Replace NaN with mean value
df['SPEEDING'].replace(np.NaN, "N", inplace=True)


# In[268]:


encoding_SPEEDING = {"SPEEDING": 
                            {"Y": 1,
                             "N": 0,
                              }}
df.replace(encoding_SPEEDING, inplace=True)
df['SPEEDING'].value_counts()


# In[269]:


#Inattentioned:
df['INATTENTIONIND'].value_counts()


# In[270]:


df['INATTENTIONIND'].replace(np.NaN, "N", inplace=True)


# In[271]:


encoding_INATTENTIONIND = {"INATTENTIONIND": 
                            {"Y": 1,
                             "N": 0,
                              }}
df.replace(encoding_INATTENTIONIND, inplace=True)
df['SPEEDING'].value_counts()


# In[272]:


#Light condition:
df['LIGHTCOND'].value_counts()


# In[273]:


#Replace NaN with "Unknown"
df['LIGHTCOND'].replace(np.NaN, "Unknown", inplace=True)


# In[274]:


encoding_LIGHTCOND = {"LIGHTCOND": 
                            {"Daylight": 0,
                             "Unknown": 0,
                             "Other": 0,
                             "Dark - Street Lights On": 1,
                             "Dusk": 1,
                             "Dawn": 1,
                             "Dark - No Street Lights": 1,
                             "Dark - Street Lights Off": 1,
                             "Dark - Unknown Lighting": 1,
                              }}
df.replace(encoding_LIGHTCOND, inplace=True)
df['LIGHTCOND'].value_counts()


# In[275]:


#Road condition:
df['ROADCOND'].value_counts()


# In[276]:


#Replace NaN with "Unknown"
df['ROADCOND'].replace(np.NaN, "Unknown", inplace=True)


# In[277]:


encoding_ROADCOND = {"ROADCOND": 
                            {"Dry": 1,
                             "Unknown": 1,
                             "Other": 1,
                             "Wet": 2,
                             "Ice": 3,
                             "Snow/Slush": 4,
                             "Standing Water": 5,
                             "Sand/Mud/Dirt": 6,
                             "Oil": 7,
                              }}
df.replace(encoding_ROADCOND, inplace=True)
df['ROADCOND'].value_counts()


# In[278]:


#Junction type:
df['JUNCTIONTYPE'].value_counts()


# In[279]:


#Replace NaN with "Unknown"
df['JUNCTIONTYPE'].replace(np.NaN, "Unknown", inplace=True)


# In[280]:


encoding_JUNCTIONTYPE = {"JUNCTIONTYPE": 
                            {"Mid-Block (not related to intersection)": 1,
                             "Unknown": 1,
                             "At Intersection (intersection related)": 2,
                             "Mid-Block (but intersection related)": 3,
                             "Driveway Junction": 4,
                             "At Intersection (but not related to intersection)": 5,
                             "Ramp Junction": 6,
                              }}
df.replace(encoding_JUNCTIONTYPE, inplace=True)
df['JUNCTIONTYPE'].value_counts()


# In[281]:


avg_LONGITUDE = df["LONGITUDE"].astype("float").mean(axis=0)
print("Average of LONGITUDE:", avg_LONGITUDE)
df['LONGITUDE'].replace(np.NaN, avg_LONGITUDE, inplace=True)


# In[282]:


avg_LATITUDE = df2["LATITUDE"].astype("float").mean(axis=0)
print("Average of LATITUDE:", avg_LATITUDE)
df['LATITUDE'].replace(np.NaN, avg_LATITUDE, inplace=True)


# In[283]:


df.isnull().sum()


# ### Balancing the dataset

# In[284]:


df["SEVERITYCODE"].value_counts().plot(kind='pie',shadow=True, startangle=90)
plt.xlabel('Severity Code (prop damage=1,injury=2)') 
plt.ylabel('Number of Accidents') 
plt.title('Car Accident Severity (Imbalanced)')


# In[285]:


# Sorting majority and minority
df_major = df[df["SEVERITYCODE"] == 1]
df_minor = df[df["SEVERITYCODE"] == 2]
df_major.shape


# In[286]:


#Balancing
df_minor_reform = resample(df_minor,
                                 replace=True,     # sample with replacement
                                 n_samples=132488,    # to match majority class
                                 random_state=250) # reproducible results

#Resampling
df2 = pd.concat([df_major, df_minor_reform])
df2.SEVERITYCODE.value_counts()


# In[287]:


df2.head()


# In[288]:


df2.columns


# ## 3. Exploratory data analysis

# In[289]:


#Installing Folium Package for mapping
get_ipython().system('conda install -c conda-forge folium=0.5.0 --yes')
import folium


# In[290]:


#VISUALIZE 400 DATA POINTS
limit = 400
df_m1 = df2[["LATITUDE", "LONGITUDE"]]
df_m2 = df_m1.iloc[0:limit, :]


# In[291]:


# INCIDENTS IN DF

Seattle_map = folium.Map(location=[47.6062, -122.3321], zoom_start=12)

incidents = folium.map.FeatureGroup()

#  400 POINTS GROUPING
for lat, lng, in zip(df_m2.LATITUDE, df_m2.LONGITUDE):
    incidents.add_child(
        folium.features.CircleMarker(
            [lat, lng],
            radius=5, # define how big you want the circle markers to be
            color='red',
            fill=True,
            fill_color='green',
            fill_opacity=0.6
        )
    )

# MAPPING
Seattle_map.add_child(incidents)


# In[292]:


# SELECT SEVERITY 2 IN NEW DF
Sev_2 = df2.loc[df2['SEVERITYCODE']==2]
Sev_2.head()


# ### Weather:

# In[293]:


Sev_2_w = Sev_2['WEATHER'].value_counts()
Sev_2_w


# In[294]:


labels = 'Clear', 'Raining', 'Overcast', 'Other'
sizes = [37856, 11176, 8745, sum(Sev_2_w[3:9])]
explode = (0.1,0, 0, 0)
fig1, ax1 = plt.subplots(figsize=(20,10))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Weather Conditions - Severity ', y=1.05)
plt.show()


# ### Person: 

# In[295]:


Sev_2_p = Sev_2['PERSONCOUNT'].value_counts()
labels = 2, 3, 4, 1, 5, 0, 6, '>6'
sizes = [27811, 13461, 6295, 3296, 2969, 1762, 1357, sum(Sev_2_p[3:9])]
explode = (0.1, 0, 0, 0, 0, 0, 0, 0)
fig1, ax1 = plt.subplots(figsize=(20,10))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Persons - Severity', y=1)
plt.show()


# ### Vehicle:

# In[296]:


Sev_2_v = Sev_2['VEHCOUNT'].value_counts()
labels = 2, 1, 3, 0, 4,'>4'
sizes = [35949, 14105, 5470, 1227, 1078, sum(Sev_2_p[5:12])]
explode = (0.1, 0, 0, 0, 0, 0)
fig1, ax1 = plt.subplots(figsize=(20,10))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Vehicles - Severity', y=1.05)
plt.show()


# ### Junction type:

# In[297]:


Sev_2_j = Sev_2['JUNCTIONTYPE'].value_counts()
labels = 'At Iintersection_intersection related', 'Mid-Block_not intersection related', 'Mid-Block with intersection', 'Driveway Junction', 'Other'
sizes = [27174, 19806, 7297, 3234, sum(Sev_2_j[4:6])]
explode = (0.1, 0, 0, 0, 0)
fig1, ax1 = plt.subplots(figsize=(20,10))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Junction Type - Severity', y=1.05)
plt.show()


# ### Inattentioned: 

# In[298]:


Sev_2_i = Sev_2['INATTENTIONIND'].value_counts()
labels = 'No', 'Yes'
sizes = [47791, 10397]
explode = (0.1, 0)
fig1, ax1 = plt.subplots(figsize=(20,10))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Inattention - Severity ', y=1)
plt.show()


# ### Road condition: 

# In[299]:


Sev_2_r = Sev_2['ROADCOND'].value_counts()

labels = 'DRY', 'WET', 'ic, sand, oil, standing water'
sizes = [41916, 15755, sum(Sev_2_r[2:7])]
explode = (0.1, 0, 0)
fig1, ax1 = plt.subplots(figsize=(20,10))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Road Conditions - Severity ', y=1.05)
plt.show()


# ### Light condition:

# In[300]:


Sev_2_l = Sev_2['LIGHTCOND'].value_counts()
labels = 'DAY LIGHT', 'DARK'
sizes = [40291, 17897]
explode = (0.1, 0)
fig1, ax1 = plt.subplots(figsize=(20,10))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Light Conditions - Severity', y=1)
plt.show()


# ### Speed: 

# In[301]:


Sev_2_s = Sev_2['SPEEDING'].value_counts()
labels = 'NO', 'YES'
sizes = [54657, 3531]
explode = (0.1, 0)
fig1, ax1 = plt.subplots(figsize=(20,10))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',shadow=True, startangle=60)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Speed - Severity', y=1)
plt.show()


# ## 4. Model development 

# In[302]:


#Defining features
Feature = df2[['LONGITUDE', 'LATITUDE', 'PERSONCOUNT', 'VEHCOUNT',
       'JUNCTIONTYPE', 'INATTENTIONIND', 'WEATHER', 'ROADCOND', 'LIGHTCOND',
       'SPEEDING']]


# In[303]:


X = Feature
y = df2['SEVERITYCODE'].values


# In[304]:


#Standardization
X= preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# In[305]:


#Train/test split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# ### KNN

# In[306]:


#Best k
Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(1,Ks):
    
    # Training 
    kNNeigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat = kNNeigh.predict(X_test)
    
    
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat);
    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc


# In[307]:


#Plot accuracy with increasing neighbours k
plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbous (K)')
plt.tight_layout()
plt.show()


# In[308]:


print( "Best k =", mean_acc.argmax()+1)
#Building model with best k == 8
k=1
kNNeigh= KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
kNNeigh


# In[309]:


#Evaluation
print("K-Nearest Neighbours Accuray: ", metrics.accuracy_score(y_test, yhat))


# ### Decision tree 

# In[310]:


#Building model
DTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
DTree.fit(X_train,y_train)


# In[311]:


#Prediction
yhat = DTree.predict(X_test)


# In[312]:


#Evaluation
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, yhat))


# ### Apply logistic regression 

# In[313]:


#Buiding model
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
LR


# In[314]:


#Predition
yhat = LR.predict(X_test)


# In[315]:


#Evaluation
print("Logistic Regresion's Accuracy: ", metrics.accuracy_score(y_test, yhat))


# ###  Random forest 

# In[316]:


#Building model
clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)


# In[317]:


#Prediction
yhat =clf.predict(X_test)


# In[318]:


#Evaluation
print("Random Forest's Accuracy: ", metrics.accuracy_score(y_test, yhat))


# ## 5. Model evaluation 

# In[319]:


algo_lst =['K-Nearest Neighbors','Decision Trees','Logistic Regression','Random Forest']

accuracy_lst = [0.7267753948889174, 0.7429562090663927, 0.7030949017593425, 0.711878772312829]

# Generate a list of ticks for y-axis
y_ticks=np.arange(len(algo_lst))

#Combine the list of algorithms and list of accuracy scores into a dataframe, sort the value based on accuracy score
df_acc=pd.DataFrame(list(zip(algo_lst, accuracy_lst)), columns=['Algorithm','Accuracy_Score']).sort_values(by=['Accuracy_Score'],ascending = True)

# Make a plot
ax=df_acc.plot.barh('Algorithm', 'Accuracy_Score', align='center',legend=False,color='0.9')

# Add the data label on to the plot
for i in ax.patches:
    # get_width pulls left or right; get_y pushes up or down
    ax.text(i.get_width()+0.02, i.get_y()+0.2, str(round(i.get_width(),2)), fontsize=10)

# Set the limit, lables, ticks and title
plt.xlim(0,1.1)
plt.xlabel('Accuracy Score')
plt.yticks(y_ticks, df_acc['Algorithm'], rotation=0)
plt.title('Model accuracy')

plt.show()


# ## 6. Conclusion

# Decision tree is the most accurate model among all model been tested in the prediction of car accident severity.
# 
# Future work will be feeding more data into the dataset to increase model accuracy. In our case, age was not considered but it could impact on driving and car accident.
