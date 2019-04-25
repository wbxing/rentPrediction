import numpy as np
import pandas as pd

# 设置控制台显示规则
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 不用科学计数法显示
np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)


# 载入训练文件
train_data = pd.read_csv("dataset/train.csv")
# 添加列名
train_data.columns = ["time", 'estate', 'rentRatio', 'height', 'heightRatio', 'area', 'orient',
                      'status', 'bedroom', 'hall', 'toilet', 'wholeRent', 'district', 'location',
                      'subwayLine', 'subwayStation', 'distance', 'decoration', 'rent']
# 载入测试文件
test = pd.read_csv("dataset/test.csv")
# 添加列名
test.columns = ["id", "time", 'estate', 'rentRatio', 'height', 'heightRatio', 'area', 'orient',
                'status', 'bedroom', 'hall', 'toilet', 'wholeRent', 'district', 'location',
                'subwayLine', 'subwayStation', 'distance', 'decoration']


# 处理训练数据
# 删除小区名为零的行
train_data = train_data[train_data.estate != 0]
# 删除总楼层为零的行
train_data = train_data[train_data.heightRatio != 0]
# 删除面积为零的行
train_data = train_data[train_data.area != 0]
# 删除租金为零的行
train_data = train_data[train_data.rent != 0]


# 填充缺省值
train_data['status'].fillna(train_data['status'].mean(), inplace=True)
train_data['rentRatio'].fillna(train_data['rentRatio'].mean(), inplace=True)
train_data['wholeRent'].fillna(train_data['wholeRent'].mean(), inplace=True)
train_data['district'].fillna(train_data['district'].mean(), inplace=True)
train_data['location'].fillna(train_data['location'].mean(), inplace=True)
train_data['subwayLine'].fillna(train_data['subwayLine'].mean(), inplace=True)
train_data['subwayStation'].fillna(train_data['subwayStation'].mean(), inplace=True)
train_data['distance'].fillna(train_data['distance'].mean(), inplace=True)
train_data['decoration'].fillna(train_data['decoration'].mean(), inplace=True)

test['status'].fillna(test['status'].mean(), inplace=True)
test['rentRatio'].fillna(test['rentRatio'].mean(), inplace=True)
test['wholeRent'].fillna(test['wholeRent'].mean(), inplace=True)
test['district'].fillna(test['district'].mean(), inplace=True)
test['location'].fillna(test['location'].mean(), inplace=True)
test['subwayLine'].fillna(test['subwayLine'].mean(), inplace=True)
test['subwayStation'].fillna(test['subwayStation'].mean(), inplace=True)
test['distance'].fillna(test['distance'].mean(), inplace=True)
test['decoration'].fillna(test['decoration'].mean(), inplace=True)


# 处理房屋朝向数据
# 替换空格为“|”
def foo(s):
    return s.replace(' ', '|')


tmp = train_data['orient'].map(foo)  # map是对每个调用foo(s)
tmp = tmp.str.get_dummies()
tmp.columns = ['east', 'northeast', 'southeast', 'north', 'south', 'west', 'northwest', 'southwest']
train = pd.concat([train_data, tmp], axis=1)
train.drop('orient', axis=1, inplace=True)
train.to_csv('dataset/train2.csv', index=False)
print(train.shape)

tmp1 = test['orient'].map(foo)
tmp1 = tmp1.str.get_dummies()
tmp1.columns = ['east', 'northeast', 'southeast', 'north', 'south', 'west', 'northwest', 'southwest']
test = pd.concat([test, tmp1], axis=1)
test.drop('orient', axis=1, inplace=True)
test.to_csv('dataset/test2.csv', index=False)
print(test.shape)
