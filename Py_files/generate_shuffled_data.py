import numpy as np
import pandas as pd

def randomize_features(input_filename, output_filename):
    data = pd.read_csv(input_filename, header=None)
    
    features = [i for i in range(1,562)] # column 0 is label, so starting from 1
    np.random.seed(18)
    np.random.shuffle(features)
    label = data.columns[0]
    
    data_features = data[features]
    data_label = data[label]
    
    result = pd.concat([data_label, data_features], axis=1)
    
    result.to_csv(output_filename, encoding='utf-8', index=False, header=None)


randomize_features('train_transformed.csv', 'train_randomized.csv')
#randomize_features('validation_transformed.csv', 'validation_randomized.csv')
randomize_features('test_transformed.csv', 'test_randomized.csv')
