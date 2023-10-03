import numpy as np  # for handling multi-dimensional array operation
import pandas as pd  # for reading data from csv 
import statsmodels.api as sm  # for finding the p-value
from sklearn.preprocessing import MinMaxScaler  # for normalization
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score 
from sklearn.utils import shuffle
# >> FEATURE SELECTION << #

#diagnosis = {'BI': 0, 'KR':1, 'LA': 2, 'LS': 3, 'PA': 4, 'UM': 5}
#f= 0 
#For each folder in our data folder:
    #for each patient:
        # d = pd.read_csv(patient file)
        # get top 20% miRNAs
        # label as active (1) and label rest of 80% as absent (0)
        # d['diagnosis'] = f
        # f++
#normalize
#train_test_split
#compute cost function
#calculate cost gradient
#minimize with sgd
#test





'''def remove_correlated_features(X):
def remove_less_significant_features(X, Y):
# >> MODEL TRAINING << #
def compute_cost(W, X, Y):
def calculate_cost_gradient(W, X_batch, Y_batch):
def sgd(features, outputs):
def init():'''