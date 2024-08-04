import os
import logging
import numpy as np
import pandas as pd
import category_encoders as ce
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.stats import randint as sp_randint
from collections import Counter

import sklearn.tree

# 读取ashrea_ii数据集
current_path = os.path.abspath(__file__)
parent_path = os.path.dirname(os.path.dirname(current_path))
ashrae_db_path = os.path.join(parent_path, 'data/archive/ashrae_db2.01.csv')
print(f"data path:{ashrae_db_path}")
assert os.path.exists(ashrae_db_path)
df = pd.read_csv(ashrae_db_path)

# 创建一个logger
logger = logging.getLogger(name="ashrae")
logger.setLevel(logging.INFO)

# 创建一个处理器
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# 创建一个格式化器
formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')
ch.setFormatter(formatter)

# 将处理器添加到记录器中
logger.addHandler(ch)

# 数据预处理-去除空值率高的字段
logger.info("clean and analysing the data:")
print(df.isnull().sum())
print(df.dtypes)

logger.info("select the columns which have less than 67% Null values:")
df = df.loc[:, df.isna().mean()<0.67]

logger.info("lets see whats all feature we have for our analysis:")
print(df.columns)

logger.info("check again:")
print(df.isnull().sum())

logger.info("check data type:")
print(df.dtypes)

# 选择特征
"""
df = df[['Publication (Citation)',
        'Data contributor',
        'Year',
        'Season',
        'Koppen climate classification',
        'Climate',
        'City',
        'Country',
        'Building type',
        'Cooling startegy_building level',
        'Sex',
        'Thermal sensation',
        'Thermal preference',
        'Thermal comfort',
        'Clo',
        'Met',
        'Air temperature (C)',
        'Relative humidity (%)',
        'Air velocity (m/s)',
        'Outdoor monthly air temperature (C)',
        'Database']]
"""
basic_identifiers = [
    'Year',     # 年度
    'Season',   # 季节
    'Climate',  # koppen季节分类
    'City',     # 城市
    'Country',  # 国家
    'Building type'     # 建筑类型
]
personal_information = [
    'Age',      # 年龄
    'Sex',      # 性别
    'Clo',      # 服装绝缘水平
    'Met'       # 平均代谢率
]
environment_features = [
    # 'Cooling startegy_building level',  # 制冷策略
    # 'Heating startegy_building level',  # 制热策略
    'Thermal sensation',        # ASHRAE热感觉投票，从-3（冷）到+3（热）
    'Thermal preference',       # 热偏好
    'Air temperature (C)',      # 气温
    'Relative humidity (%)',    # 相对湿度
    'Air velocity (m/s)',       # 空气流速
    'Outdoor monthly air temperature (C)',      # 室外月平均气温
]
label = ['Thermal comfort']

logger.info("select the needed colunms:")
df = df[basic_identifiers + personal_information + environment_features + label]
print(df.dtypes)
data_type = df.dtypes
print(f"type of sex: {data_type['Sex']}")
print(f"type {type(data_type)}")
if data_type['Sex'] == 'object':
    print("-===")
logger.info("remove na:")
df = df.dropna()
print(df.isnull().sum())

logger.info("check for outliers")
print(type(df))
df = df[df["Thermal comfort"]!='Na']
df = df[df["Thermal comfort"]!=1.3]

df['Thermal comfort'] = df['Thermal comfort'].astype('int64')
df['Thermal comfort'].value_counts()


logger.info("convert to categorical data type")
df['Thermal comfort'] = df['Thermal comfort'].astype('category')
print(df.dtypes)

logger.info("convert data type of all columns with 'objects' to 'category':")
df = pd.concat([df.select_dtypes([], ['object']), df.select_dtypes(["object"]).apply(pd.Series.astype, dtype='category')], axis=1)
print(df.dtypes)

# splitting the Data
y = df['Thermal comfort']
df.drop(['Thermal comfort'], axis=1, inplace=True)

# one hot encoding
# 定义one-hot编码特征和不需要编码的特征
categorical_features = ['Season', 'Climate', 'City', 'Country', 'Building type', 'Sex', 'Thermal preference']
numerical_features = ['Year', 'Age', 'Clo', 'Met', 'Thermal sensation', 'Air temperature (C)', 'Relative humidity (%)', 'Air velocity (m/s)', 'Outdoor monthly air temperature (C)']
# 创建预处理管道
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numerical_features)
    ]
)

# 定义pipline
transformers = Pipeline(steps=[
    # ('Thermal_comfort', OneHotEncoder(), ['Thermal_comfort']),
    ('Season', OneHotEncoder(), ['Season']),
    ('Climate', OneHotEncoder(), ['Climate']),
    ('City', OneHotEncoder(), ['City']),
    ('Country', OneHotEncoder(), ['Country']),
    ('Building_type', OneHotEncoder(), ['Building type']),
    ('Sex', OneHotEncoder(), ['Sex']),
    ('Thermal_preference', OneHotEncoder, ['Thermal preference'])
])
# 定义转换器
# transformer = ColumnTransformer(transformers=transformers)
# df = transformer.fit_transform(df)
# df = pd.get_dummies(data=df, drop_first=True)


# splitting to train and test set
x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state= 42)


# Random Forest
clf = RandomForestClassifier(n_estimators=540)
# 将预处理管道和回归模型放入整体管道
clf_pipleline = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('regressor', clf)
])
clf_pipleline.fit(x_train, y_train)
pred = clf_pipleline.predict(x_test)
print("Accuracy of Random Forest:%4f,mse:%4f"%(accuracy_score(y_test, pred)*100, mean_squared_error(y_test, pred)))
confusion_matrix(y_test, pred)

# df = pd.get_dummies(data=df, drop_first=True)
# x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state= 42)
# Naive Bayes Classification Algorithm
glf = GaussianNB()
glf_pipeline = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('regressor', glf)
])
glf_pipeline.fit(x_train, y_train)
pred = glf_pipeline.predict(x_test)
print("Accuracy of Naive Bayes Classificatin %4f,mse:%4f"%(accuracy_score(y_test, pred)*100, mean_squared_error(y_test, pred)))
confusion_matrix(y_test, pred)


# Decision tree classifier
dlf = sklearn.tree.DecisionTreeClassifier()
dlf_pipeline = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('regressor', dlf)
])
dlf_pipeline.fit(x_train, y_train)
pred = dlf_pipeline.predict(x_test)
print("Accuracy of decision tree %4f,mse:%4f"%(accuracy_score(y_test, pred)*100, mean_squared_error(y_test, pred)))
confusion_matrix(y_test, pred)

# baggingClassifier
blf = BaggingClassifier(n_estimators=540)
blf_pipeline = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('regressor', blf)
])
blf_pipeline.fit(x_train, y_train)
pred = blf_pipeline.predict(x_test)
print("Accuracy of bagging Classificatin %4f,mse:%4f"%(accuracy_score(y_test, pred)*100, mean_squared_error(y_test, pred)))
confusion_matrix(y_test, pred)

# gradient
"""
# 随机搜索参数
glf_param_dist = {
    "n_estimators": sp_randint(50, 200),
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth": sp_randint(3, 6),
    "min_samples_split": sp_randint(2, 11),
    "min_samples_leaf": sp_randint(1, 5),
    "subsample": [0.8, 1.0]
}
# 实例化模型
glf = GradientBoostingClassifier(random_state=42)
# 实例化randomcv
glf_random_search = RandomizedSearchCV(estimator=glf, param_distributions=glf_param_dist, n_iter=100, cv=5, n_jobs=1, verbose=2, random_state=42)
# 拟合模型
glf_random_search.fit(x_train, y_train)
glf_best = glf_random_search.best_estimator_
pred = glf_best.predict(x_test)
print("best parameters found: ", glf_random_search.best_params_)
# best parameters found:  {'learning_rate': 0.01, 'max_depth': 5, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 184, 'subsample': 0.8}
"""


glf = GradientBoostingClassifier(loss='log_loss', learning_rate=0.1, n_estimators=200, max_depth=5, min_samples_split=2, verbose=1)
glf_pipeline = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('regressor', glf)
])
glf_pipeline.fit(x_train, y_train)
pred = glf_pipeline.predict(x_test)
sns.heatmap(confusion_matrix(y_test, pred), annot=True, robust=True)
plt.show()
print("Accuracy of gradient boosting %4f,mse:%4f"%(accuracy_score(y_test, pred)*100, mean_squared_error(y_test, pred)))
