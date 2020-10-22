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


results = []
perplexities = [10, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210, 230, 250, 270, 290, 310, 330, 350]
#perplexities = [10, 30]
for i in perplexities:
    temp = []
    train_filename = "train_randomized_js_" + str(i) + ".csv"
    test_filename = "test_randomized_js_" + str(i) + ".csv"
    y_train, X_train = separate_feature_label(train_filename)
    y_test, X_test = separate_feature_label(test_filename)
    rf_result = randomForest(X_train, y_train, X_test, y_test)
    temp.append(i)
    temp.append(rf_result)
    results.append(temp)
results = np.transpose(results)
results_pd = pd.DataFrame(results)
results_pd.to_csv('rf_js_har1d.csv', encoding='utf-8', index=False, header=None)


'''
fileName = "rf_" + ".csv"
with open(fileName,'w+') as out_file:
    print(rf_results, file=out_file)
    print(rf_avg, file=out_file)
'''