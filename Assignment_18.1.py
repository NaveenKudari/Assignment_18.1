
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
boston = datasets.load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = boston.target
#X = boston.drop(['Name', 'Ticket', 'Cabin', 'PassengerId','Survived','Embarked'], axis=1)
X.shape #(506,13)
y.shape #(506,)
scaler = StandardScaler().fit(X)
#print(scaler)
rescaledX = scaler.transform(X)
#print(rescaledX)
#print(y)
#rescaledX = rescaledX.reshape(506,13,-1)
#rescaledX = rescaledX.tranpose()
#rescaledX=rescaledX.reshape(1, -1)
X_train, X_test, y_train, y_test = train_test_split(rescaledX, y, random_state=3)
rfc1 = RandomForestRegressor(max_features=5, random_state=1)
rfc1.fit(X_train, y_train)
pred1 = rfc1.predict(X_test)
print(pred1)
#print(accuracy_score(y_test, pred1))
#rfc1.score(y_test, pred1)

