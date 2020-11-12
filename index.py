import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

# make it so that the console prints full DataFrames and tables
desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 15)

data = pd.read_csv('data.csv')

num_classes = 46
img_width = 32
img_height = 32
img_depth = 1

X = np.array(data.iloc[:, :-1])
Y_d = np.array(data.iloc[:, -1])

binary_encoder = LabelBinarizer()
y = binary_encoder.fit_transform(Y_d)
X = X / 255.0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)

X_test, X_dev, y_test, y_dev = train_test_split(X_test, y_test, test_size=0.5, random_state=101)

# 1024 -> 896 -> 632 -> 372 -> 46
print('Starting')
model = Sequential()
model.add(Dense(896, input_dim=1024, activation='relu'))
model.add(Dense(632, activation='relu'))
model.add(Dense(372, kernel_regularizer=tf.keras.regularizers.l2(0.00005), activation='relu'))
model.add(Dense(46, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=40, batch_size=1750)

print('Done traning, testing on dev set')
# evaluate the keras model

_, accuracy = model.evaluate(X_dev, y_dev)
print('Accuracy on dev set: %.2f' % (accuracy*100))

print('Done testing on the dev set, testing on test set')

_, accuracy = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy*100))
