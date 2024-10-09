import numpy as np
import pandas as pd
from pyts.image import GramianAngularField, MarkovTransitionField

# 加载数据
def load_data(file_path):
    data = pd.read_csv(file_path)
    X_train = data.drop(columns=['subject', 'label'], axis=1)
    X_train = X_train.iloc[:, 1:]
    Y_train = data.loc[:, ['label']].squeeze()
    return X_train, Y_train

# 分位数归一化
def quantile_normalize(df):
    df_sorted = pd.DataFrame(np.sort(df.values, axis=0), index=df.index, columns=df.columns)
    df_mean = df_sorted.mean(axis=1)
    df_mean.index = np.arange(1, len(df_mean) + 1)
    df_qn = df.rank(method="min").stack().astype(int).map(df_mean).unstack()
    return df_qn

# 将时间序列转换为图像
def transform_to_images(X_train):
    gasf = GramianAngularField(method='summation')
    gadf = GramianAngularField(method='difference')
    mtf = MarkovTransitionField()

    X_train = X_train.to_numpy()
    X_train_gasf = gasf.fit_transform(X_train)
    X_train_gadf = gadf.fit_transform(X_train)
    X_train_mtf = mtf.fit_transform(X_train)

    X_train = np.stack((X_train_gasf, X_train_gadf, X_train_mtf), axis=-1)
    return X_train

# 预处理主函数
def preprocess_data(file_path):
    X_train, Y_train = load_data(file_path)
    X_train = quantile_normalize(X_train)
    X_train = transform_to_images(X_train)
    return X_train, Y_train
