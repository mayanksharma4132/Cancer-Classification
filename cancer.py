#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


#getting data and storing in dataset variable df using pandas
df =pd.read_csv('data.csv')
#print(df.info())
print(df.columns)

#getting all the feature colums in x
x =df.drop(["id","diagnosis"],axis=1)

#replacing missing values with mean

imputer = Imputer(missing_values = 'NaN',strategy= 'mean' , axis = 0)
imputer = imputer.fit(x)
x = imputer.transform(x)
#gettng the result column in y
y =df["diagnosis"]

#converting categorical data into indicator/dummy variable
y = pd.get_dummies(df["diagnosis"],drop_first=True)
print(df.head())

#Splitting the data into 0.8 train and 0.2 test
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

#scalling features
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
