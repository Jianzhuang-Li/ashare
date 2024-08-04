import os
import logging
import numpy as np
import pandas as pd
import category_encoders as ce
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.tree
import json
from typing import Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler



# 读取ashrea_ii数据集
current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
ashrae_db_path = os.path.join(current_dir, 'data/archive/ashrae_db2.01.csv')
print(f"data path:{ashrae_db_path}")
assert os.path.exists(ashrae_db_path)
df_raw = pd.read_csv(ashrae_db_path)

# ashrea_ii数据集的数据特征



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

# 配置文件路径
default_config:str =  os.path.join(current_dir, 'config/default_config.json')
thermal_comfort_config:str = os.path.join(current_dir, 'config/thermal_comfort_config.json')


def ReadConfig(conf_path:str=default_config)->Any:
    """
    Read thermal prediction model config.
    """
    print(conf_path)
    assert os.path.exists(conf_path)
    with open(conf_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def GetFeaturesAndLable(conf_path:str, df):
    # 获取配置
    default_config = ReadConfig()
    regress_config = ReadConfig(conf_path=conf_path)
    
    # 源数据特征
    basic_identifiers:set[str] = set(default_config["basic_identifiers"])
    personal_information:set[str] = set(default_config["personal_information"])
    thermal_comfor_info:set[str] = set(default_config["thermal_comfor_info"])
    thermal_confort_measurements:set[str] = set(default_config["thermal_confort_measurements"])
    calculated_indices:set[str] = set(default_config["calculated_indices"])
    environment_control:set[str] = set(default_config["environment_control"])
   
    # 用于回归预测的数据特征
    basic_identifiers_u:set[str] = set(regress_config["basic_identifiers"])
    personal_information_u:set[str] = set(regress_config["personal_information"])
    thermal_comfor_info_u:set[str] = set(regress_config["thermal_comfor_info"])
    thermal_confort_measurements_u:set[str] = set(regress_config["thermal_confort_measurements"])
    calculated_indices_u:set[str] = set(regress_config["calculated_indices"])
    environment_control_u:set[str] = set(regress_config["environment_control"])
    
    # 数据特征检查
    assert basic_identifiers_u.issubset(basic_identifiers)
    assert personal_information_u.issubset(personal_information)
    assert thermal_comfor_info_u.issubset(thermal_comfor_info)
    assert thermal_confort_measurements_u.issubset(thermal_confort_measurements)
    assert calculated_indices_u.issubset(calculated_indices)
    assert environment_control_u.issubset(environment_control)

    # 合并数据特征，获取标签
    features = basic_identifiers_u | personal_information_u | thermal_comfor_info_u \
        | thermal_confort_measurements_u | calculated_indices_u | environment_control_u
    features = list(features)
    labels = [regress_config["label"]]

    # 去除空值率高的的字段后，检查使用的数据特征是否都还存在
    colums_names = df.columns
    for feature in features:
        print(feature)
        assert feature in colums_names
    for label in labels:
        assert label in colums_names

    feature_label_map = {
        "features": features,
        "labels": labels
    }

    return feature_label_map

def DataPreproess(conf_path, df):
    regress_config = ReadConfig(conf_path)
    # 数据预处理-去除空值率高的字段
    max_nan_rate:float = regress_config['max_nan_rate']
    logger.info("clean and analysing the data:")
    print(df.isnull().sum())

    logger.info("select the columns which have less than 67% Null values:")
    df = df.loc[:, df.isna().mean()<0.67]

    logger.info("lets see whats all feature we have for our analysis:")
    print(df.columns)

    logger.info("check again:")
    print(df.isnull().sum())

    # 获取特征列和标签列
    feature_label_map = GetFeaturesAndLable(conf_path, df)
    features = feature_label_map["features"]
    labels = feature_label_map["labels"]
    df = df[features+labels]
   
     # 去除空值行
    logger.info("remove na:")
    df = df.dropna()
    print(df.isnull().sum())

    logger.info("lets see whats all feature we have for our analysis:")
    print(type(df))
    print(df.columns)

    # 单独处理Thermal comfort
    if "Thermal comfort" in df.columns:
        logger.info("check for outliers")
        df = df[df["Thermal comfort"]!='Na']
        df = df[df["Thermal comfort"]!=1.3]
        df['Thermal comfort'] = df['Thermal comfort'].astype('int64')
        df['Thermal comfort'] = df['Thermal comfort'].astype('category')
        df['Thermal comfort'].value_counts()

    # object类型转换为category
    logger.info("convert data type of all columns with 'objects' to 'category':")
    # df = pd.concat([df.select_dtypes([], ['object']), df.select_dtypes(["object"]).apply(pd.Series.astype, dtype='category')], axis=1)
    print("b dp")
    print(df.count())
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category')
    print(df.dtypes)

    preprocess_map = {
        "data": df,
        "features": features,
        "labels": labels
    }

    return preprocess_map
   

def GetPipelineFeature(conf_path:str, df):
    # 获取特征和标签
    feaure_lable_map = DataPreproess(conf_path, df)
    labels = feaure_lable_map["labels"]
    features = feaure_lable_map["features"]
    df = feaure_lable_map["data"]
    print("In get piprline feature")
    print(df.count())

    # 获取数据特征
    data_type = df.dtypes

    # 定义one-hot编码特征和不需要编码的特征
    categorical_features = []
    numerical_features = []

    for feature in features:
        if data_type[feature] == 'category':
            categorical_features.append(feature)
        elif data_type[feature] == 'float64':
            numerical_features.append(feature)
        else:
            raise TypeError(f"Unsuport type: {df[feature].dtype} of {feature}")
    pipelie_features = {
        "labels": labels,
        "numerical_features": numerical_features,
        "category_features": categorical_features,
        "data": df
    }
    return pipelie_features

if __name__ == "__main__":

    # df = DataPreproess(thermal_comfort_config, df)
    pipeline_feature = GetPipelineFeature(thermal_comfort_config, df_raw)


    # one hot encoding
    # 定义one-hot编码特征和不需要编码的特征
    labels = pipeline_feature["labels"]
    categorical_features = pipeline_feature["category_features"]
    numerical_features = pipeline_feature["numerical_features"]
    df = pipeline_feature["data"]
    print(df.count())
    # splitting the Data
    x = df[categorical_features + numerical_features]
    y = df[labels]
    # 创建预处理管道
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', StandardScaler(), numerical_features)
        ]
    )

    # splitting to train and test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state= 42)


    glf = GradientBoostingClassifier(loss='log_loss', learning_rate=0.1, n_estimators=200, max_depth=5, min_samples_split=2, verbose=1)
    glf_pipeline = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('regressor', glf)
    ])

    glf_pipeline.fit(x_train, np.ravel(y_train))
    pred = glf_pipeline.predict(x_test)
    sns.heatmap(confusion_matrix(y_test, pred), annot=True, robust=True)
    plt.show()
    print("Accuracy of gradient boosting %4f,mse:%4f"%(accuracy_score(y_test, pred)*100, mean_squared_error(y_test, pred)))
