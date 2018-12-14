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

X_train, X_test, y_train, y_true = train_test_split(train_X, train_Y, test_size=0.2)

true_list = y_true.tolist()

# 参数调优
params = [1800, 2000, 2200, 2500, 2800, 3000]
for param in params:
    print(param)
    model = xgb.XGBRegressor(n_estimators=param,
                             learning_rate=0.05,
                             max_depth=11,
                             min_child_weight=1,
                             gamma=0.05,
                             seed=0,
                             subsample=0.8,
                             colsample_bytree=0.8,
                             reg_alpha=0,
                             reg_lambda=1
                             )
    model.fit(X_train, y_train)
    y_pre = model.predict(X_test)
    y_pre_list = y_pre.tolist()
    s = 0
    for i in range(len(true_list)):
        s += ((y_pre_list[i] - true_list[i])**2)
    mse = s / len(true_list)
    rmse = math.sqrt(mse)
    print(rmse)
