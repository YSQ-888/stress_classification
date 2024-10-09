from transform import preprocess_data
X_train, Y_train = preprocess_data('../may14_feats4.csv')

from train_model import train_model
from keras.utils import to_categorical

Y_train = to_categorical(Y_train, num_classes=3)
model, history, x_test, y_test = train_model(X_train, Y_train)

from test_model import evaluate_model, plot_training

evaluate_model(model, x_test, y_test)
plot_training(history)
