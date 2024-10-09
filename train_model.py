import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, BatchNormalization, Flatten, Dense
from keras import backend as K


# 定义精度、召回率和F1分数
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


# CNN模型构建
def model_CNN():
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(7, 7), activation='relu', input_shape=(46, 46, 3)))
    model.add(MaxPool2D(3, 2))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(2, 2))
    model.add(Flatten())
    model.add(Dense(6))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy', f1_m, precision_m, recall_m])
    return model


# 模型训练
def train_model(X_train, Y_train):
    x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.3)
    model = model_CNN()

    history = model.fit(x_train, y_train, batch_size=32, epochs=70, validation_data=(x_test, y_test))

    return model, history, x_test, y_test
