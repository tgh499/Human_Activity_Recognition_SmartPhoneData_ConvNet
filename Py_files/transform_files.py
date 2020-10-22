import pandas as pd

def transform_file(filename, output_filename):
    data = pd.read_csv(filename, header=None)
    features= data.columns[1:]
    label = data.columns[0]
    data_features = data[features]
    data_label = data[label]
    data_features = data_features.add(1)
    data_features = data_features.mul(127.5)
    data_merged = pd.concat([data_label, data_features], axis=1)
    data_merged.to_csv(output_filename, encoding='utf-8', index=False, header=None)

transform_file('test.csv', 'test_transformed.csv')
#transform_file('validation.csv', 'validation_transformed.csv')
transform_file('train.csv', 'train_transformed.csv')