import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import math


# 这里读入的是处理之后的数据
train = pd.read_csv("dataset/train2.csv")
test = pd.read_csv("dataset/test2.csv")

train_X = train.drop('rent', axis=1)
train_Y = train.loc[:, 'rent']

model = XGBRegressor(learning_rate=0.01,
                     n_estimators=2800,
                     max_depth=12,
                     gamma=0.05,
                     min_child_weight=1,
                     seed=0,
                     subsample=0.8,
                     colsample_bytree=0.8,
                     reg_alpha=0,
                     reg_lambda=1)
# train_X = train_X.drop('time', axis=1)
model.fit(train_X, train_Y)
X = test.drop('id', axis=1)
# X = X.drop('time', axis=1)
Y = model.predict(X)

ans = pd.DataFrame(Y, columns=['price'])
test_id = pd.read_csv("dataset/test2.csv")
pd.concat([test_id['id'], ans], axis=1).to_csv('ans/xgb_ans.csv', index=False)
