#!/ddn/home4/r2444/anaconda3/bin/python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random

def rearrange_feature_indices(perplexity, data_features):

    mapping_filename = "mapping_ecl_" + str(perplexity) + ".csv"
    mapping = pd.read_csv(mapping_filename, header=None)
    mapping = mapping.values

    if np.min(mapping) < 0:
        mapping = mapping - np.min(mapping)

    mapping_dict = {}
    #print(mapping)
    for i,j in enumerate(mapping):
        mapping_dict[j[0]] = i
    #print(mapping)
    mapping_keys_sorted = sorted(mapping_dict.keys())
    #print(mapping_keys_sorted)

    oneD_mapping = []
    for i in mapping_keys_sorted:
        oneD_mapping.append(mapping_dict[i])
    #print(oneD_mapping)
    dim = 561

    tsne_mapped_data_features = np.zeros((len(data_features), dim))

    for i,j in enumerate(tsne_mapped_data_features):
        for k,l in enumerate(oneD_mapping):
            tsne_mapped_data_features[i][k] = data_features[i][l]
    
    return(tsne_mapped_data_features)


def reorganize_dataset(input_filename_prefix):
    input_filename = input_filename_prefix + ".csv"
    data = pd.read_csv(input_filename, header=None)

    features= data.columns[1:]
    label = data.columns[0]
    data_features = data[features]
    data_label = data[label]
    data_features_np = data_features.to_numpy()
    data_label_np = data_label.to_numpy()

    perplexities = [10, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210, 230, 250, 270, 290, 310, 330, 350]
    #perplexities = [110]
    for perplexity in perplexities:
        output_filename = input_filename_prefix + "_ecl_"+ str(perplexity) + ".csv"
        data_features_tsne_mapped = rearrange_feature_indices(perplexity, data_features_np)
        data_features_tsne_mapped_pd = pd.DataFrame(data_features_tsne_mapped)

        result = pd.concat([data_label, data_features_tsne_mapped_pd], axis=1)
        result.to_csv(output_filename, encoding='utf-8', index=False, header=None)


reorganize_dataset('test_randomized')
reorganize_dataset('train_randomized')
