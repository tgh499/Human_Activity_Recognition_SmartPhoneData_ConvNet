#!/usr/bin/env python

## Random Forest on Permuted MNIST data. Acccuracy ~ 97.4%.

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

def randomForest(X_train, y_train, X_test, y_test):
    rf = RandomForestClassifier(verbose=1)
    rf.fit(X_train, y_train)
    y_predict = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_predict)
    return(accuracy)


train = pd.read_csv('train.csv', header=None)
features_train = train.columns[1:]
labels_train = train.columns[0]
X_train = train[features_train]
y_train = train[labels_train]

test = pd.read_csv('test.csv')
features_test = test.columns[1:]
labels_test = test.columns[0]
X_test = test[features_test]
y_test = test[labels_test]


rf_results = []
for i in range(1):
    temp0 = randomForest(X_train, y_train, X_test, y_test)
    rf_results.append(temp0)

rf_avg = np.mean(rf_results)
print(rf_avg)

'''
fileName = "rf_" + str(sample_size) + ".csv"
with open(fileName,'a+') as out_file:
    print(rf_results, file=out_file)
    print(rf_avg, file=out_file)
'''