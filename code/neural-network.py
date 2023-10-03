from unittest.util import _MIN_COMMON_LEN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.python import keras
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from tensorflow.python.keras import utils
from sklearn.preprocessing import OneHotEncoder

tf.config.optimizer.set_jit(True)
tf.function(jit_compile=True)

df = []

data_dir = os.path.join(os.getcwd(), "data")
bad_files = ['MANIFEST.txt', 'annotations.txt', '.DS_Store']
for root, dirs, files in os.walk(data_dir):
    path = os.path.dirname(root)
    p = os.path.basename(path)
    for file in files:
        if file in bad_files:
            continue
        else:
            if ".txt" in file:
                pathToFile = os.path.join(root, file)
                mRNA_profile = pd.read_csv(pathToFile, sep="\t")
                x = pd.pivot_table(mRNA_profile,
                                   values='reads_per_million_miRNA_mapped',
                                   columns=['miRNA_ID'])
                x["Class"] = p
                df.append(x)

cancer_data = pd.concat(df)

Y = cancer_data['Class']
X = cancer_data.drop('Class', axis=1)

# ensure all data are floating point values
X = X.astype('float32')
# encode strings to integer
Y = LabelEncoder().fit_transform(Y)
integerEnc = Y.reshape(len(Y), 1)
onehot_encoder = OneHotEncoder(sparse=False)
Y = onehot_encoder.fit_transform(integerEnc)
bad_x = X.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(bad_x)
X = pd.DataFrame(x_scaled)
X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.76,
                                                    random_state=1)

n_features = X_train.shape[1]

c_network = Sequential()
c_network.add(Flatten(input_shape=(n_features, )))
c_network.add(Dense(80, activation="relu"))
c_network.add(Dense(15, activation="relu"))
c_network.add(Dense(6, activation="softmax"))

c_network.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

history = c_network.fit(X_train,
                        Y_train,
                        epochs=80,
                        batch_size=33,
                        validation_split=0.23)

hist_df = pd.DataFrame(history.history)

pred = c_network.predict(X_test)

plt.subplot(211)
plt.plot(hist_df['loss'])
plt.plot(hist_df['val_loss'])
plt.title('Model Metrics')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')

plt.subplot(212)
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.plot(hist_df['accuracy'], label='train')
plt.plot(hist_df['val_accuracy'], label='test')
plt.legend()
plt.show()