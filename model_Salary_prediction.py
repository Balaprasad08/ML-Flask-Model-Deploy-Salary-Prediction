import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os

df=pd.read_csv('hiring.csv')
df.head()
df.shape
df.isnull().sum()
df.experience=df.experience.fillna(0)
df.isnull().sum()
df['test_score(out of 10)']=df['test_score(out of 10)'].fillna(df['test_score(out of 10)'].median())
df.isnull().sum()
df
exp={'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,'eleven':11,0:0}
df.experience=df.experience.map(exp)
df
X=df.iloc[:,:-1]
X
y=df.iloc[:,-1]
y
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
X_train.shape,X_test.shape,y_train.shape,y_test.shape
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
y_pred
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,accuracy_score,classification_report
print('MAE',mean_absolute_error(y_test,y_pred))
print('MSE',mean_squared_error(y_test,y_pred))
print('RMSE',np.sqrt(mean_absolute_error(y_test,y_pred)))
print('Accuracy_score',r2_score(y_test,y_pred))
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
def check_model(model,X_train,X_test,y_train,y_test):
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    print('MAE',mean_absolute_error(y_test,y_pred))
    print('MSE',mean_squared_error(y_test,y_pred))
    print('RMSE',np.sqrt(mean_absolute_error(y_test,y_pred)))
    print('Accuracy_score',r2_score(y_test,y_pred))
check_model(DecisionTreeRegressor(),X_train,X_test,y_train,y_test)
check_model(RandomForestRegressor(),X_train,X_test,y_train,y_test)
check_model(SVR(),X_train,X_test,y_train,y_test)
check_model(LinearRegression(),X_train,X_test,y_train,y_test)
model.score(X_train,y_train)
r2_score(y_test,y_pred)
y_test
y_pred
model.score(X_train,y_train)
r2_score(y_test,y_pred)
import pickle
pickle.dump(model,open('model.pkl','wb'))
m=pickle.load(open('model.pkl','rb'))
m.predict(X_test)
m.score(X_train,y_train)
y_pred=m.predict(X_test)