import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

def randomForest(X_train, y_train, X_test, y_test):
    rf = RandomForestClassifier(n_estimators=750, max_features=40,
        bootstrap=False, n_jobs=-1, verbose=1, random_state=1988)
    rf.fit(X_train, y_train)
    y_predict = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_predict)
    return(accuracy)

def separate_feature_label(filename):
    data = pd.read_csv(filename)
    data_feature_col = data.columns[1:]
    data_label_col = data.columns[0]
    data_features = data[data_feature_col]
    data_label = data[data_label_col]
    return(data_label, data_features)

train_filename = "train_randomized.csv"
test_filename = "test_randomized.csv"
y_train, X_train = separate_feature_label(train_filename)
y_test, X_test = separate_feature_label(test_filename)
rf_result = randomForest(X_train, y_train, X_test, y_test)
print(100 - rf_result*100)

# original = 7.976917854718252
# randomized = 7.739307535641544


'''
fileName = "rf_" + ".csv"
with open(fileName,'w+') as out_file:
    print(rf_results, file=out_file)
    print(rf_avg, file=out_file)
'''