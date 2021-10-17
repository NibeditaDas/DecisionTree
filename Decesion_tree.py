import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

def read_data():
    data = pd.read_csv('C:\\Sudarshan\\ds_training\\Datasets\\Decision_tree_train.csv')
    return data

def null_impute(data):
    data['Age'] = data.groupby("Sex")['Age'].transform(lambda x: x.fillna(x.mean())) 
    return data

#changing Categorical column to Numerical  Column 
def label_enc(data):
    categorical_enc = LabelEncoder()
    data['Sex']=categorical_enc.fit_transform(data['Sex'])
    return data

#Model

def des_tree(X_train,y_train):
    ds_tree = DecisionTreeClassifier(max_depth= 5)
    ds_tree.fit(X_train,y_train)
    print('Accuracy' , ds_tree.score(X_train,y_train))




df = read_data()
print(df.head(5))
print(df.dtypes)
print(df.columns)
print(df.size)
print(df.shape)
print(df.isnull().sum())
null_value= null_impute(df)
print(null_value.isnull().sum())
df = df[['Survived','Pclass','Sex','Age','Fare']]
df=label_enc(df)
print(df)
X = df.drop('Survived',axis = 1)
y = df['Survived']
X_train,x_test,y_train,y_test = train_test_split(X,y,train_size=0.2,random_state=42)
dt=des_tree(X_train,y_train)
print(dt)


