import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


# 评估模型并绘制混淆矩阵
def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))

    # 归一化并绘制混淆矩阵
    matrix_float = matrix.astype(float)
    matrix_rounded = np.round(matrix_float / matrix_float.sum(axis=1, keepdims=True), decimals=4)

    sns.heatmap(matrix_rounded, cmap='Blues', annot=True, fmt=".4f")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # 输出分类报告
    y_test_wesad = y_test.argmax(axis=1)
    print(classification_report(y_test_wesad, y_pred.argmax(axis=1), target_names=["Class 0", "Class 1", "Class 2"],
                                digits=4))


# 绘制训练曲线
def plot_training(history):
    plt.plot(history.history['val_accuracy'])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig('../picture/accuracy.png')
    plt.show()

    plt.plot(history.history['loss'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('../picture/loss.png')
    plt.show()
