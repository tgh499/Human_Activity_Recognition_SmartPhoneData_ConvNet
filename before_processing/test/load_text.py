import numpy as np
import pandas as pd

data = open('X_test.txt','r')
data = data.readlines()

data_features = []
for items in data:
    items_split = items.split()
    items_split = [float(k) for k in items_split if float(k)]
    data_features.append(items_split)

data_labels_txt = open('Y_test.txt','r')
data_labels = [int(l) for l in data_labels_txt.readlines()]

data_labels_pd = pd.DataFrame(data_labels)
data_features_pd = pd.DataFrame(data_features)
data_pd = pd.concat([data_labels_pd, data_features_pd], axis=1)
data_pd.fillna(0, inplace = True)
data_pd.to_csv('test.csv', encoding='utf-8', index=False, header=None)