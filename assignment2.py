# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 23:42:51 2017

@author: user
"""

import pandas as pd
train1 = pd.read_csv("D:\Dummy.csv")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

train2=train1.loc[train1['ShearTypeClass']==0]
train, test = train_test_split(train2, test_size = 0.2)
feature1 = train.ix[:,6:20]
target1 = train["alpha"].values

feature2 = test.ix[:,6:20]
target2= test["alpha"].values

regr_1=DecisionTreeRegressor(max_depth=2)
regr_2=DecisionTreeRegressor(max_depth=5)
regr_3=DecisionTreeRegressor(max_depth=15)
regr_4=DecisionTreeRegressor(max_depth=25)
regr_5=DecisionTreeRegressor(max_depth=50)

regr_1.fit(feature1,target1)
regr_2.fit(feature1,target1)
regr_3.fit(feature1,target1)
regr_4.fit(feature1,target1)
regr_5.fit(feature1,target1)

crossvalidation=KFold(n=feature1.shape[0],n_folds=5,shuffle=True,random_state=1)
score=np.mean(cross_val_score(regr_3,feature1,target1,scoring='mean_squared_error',cv=crossvalidation,n_jobs=1))
print(score)

y_1=regr_1.predict(feature2)
y_2=regr_2.predict(feature2)
y_3=regr_3.predict(feature2)
y_4=regr_4.predict(feature2)
y_5=regr_5.predict(feature2)

s1=r2_score(target2,y_1)
s2=r2_score(target2,y_2)
s3=r2_score(target2,y_3)
s4=r2_score(target2,y_4)
s5=r2_score(target2,y_5)
print(s1)
print(s2)
print(s3)
print(s4)
print(s5)

alpha=0.001
lasso=Lasso(alpha=alpha)
y_pred_lasso=lasso.fit(feature1,target1).predict(feature2)
r2_score_lasso=r2_score(target2,y_pred_lasso)
print(r2_score_lasso)

enet=ElasticNet(alpha=alpha,l1_ratio=0.7)
y_pred_enet=enet.fit(feature1,target1).predict(feature2)
r2_score_enet=r2_score(target2,y_pred_enet)
print(r2_score_enet)

#np.savetxt('target12.csv',target2)
#np.savetxt('y11.csv',y_1)
#np.savetxt('y12.csv',y_2)


#z=range(1270)
#plt.figure()
#plt.scatter(z,y_1,color="cornflowerblue",label="max_depth=2",linewidth=2)
#plt.show()