from unittest.util import _MIN_COMMON_LEN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.python import keras
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

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
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(X)
X = pd.DataFrame(x_scaled)
X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.68,
                                                    random_state=1)

model = KNeighborsClassifier(n_neighbors=26)

model.fit(X_train, Y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import classification_report

print("EVALUATION ON TESTING DATA WITH K=26")
print(classification_report(Y_test, y_pred))

from sklearn.metrics import confusion_matrix

print("CONFUSION MATRIX")
confusion = confusion_matrix(Y_test, y_pred)
print(confusion)