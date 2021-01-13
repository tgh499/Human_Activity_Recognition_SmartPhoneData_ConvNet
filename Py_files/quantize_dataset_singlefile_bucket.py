#!/ddn/home4/r2444/anaconda3/bin/python
import pandas as pd
import numpy as np
from sklearn.datasets import load_sample_image
from sklearn.feature_extraction import image
import copy
import math

def entropy(signal):
    '''
    function returns entropy of a signal
    signal must be a 1-D numpy array
    '''
    signal = np.array(signal)
    lensig=signal.size
    symset=list(set(signal))
    #numsym=len(symset)
    propab=[np.size(signal[signal==i])/(1.0*lensig) for i in symset]
    #print(propab)
    ent= np.sum([p* math.log(1.0/p, 17) for p in propab])
    #print(ent)
    return (ent)

def get_entropy_sample_from_patch_entropy(image_sample):
    #image_sample_normalized = np.true_divide(image_sample, 255)
    new_image_sample = []
    ranges = list(range(0, 256, 15))
    range_dict = {}
    for item,value in enumerate(ranges):
        range_dict[value] = item

    for index, feature in enumerate(image_sample):
        for item in ranges:
            if feature <= item:
                new_image_sample.append(range_dict[item])
                break

    N = 4
    patches = []

    for col in range(len(new_image_sample)):
        Lx=np.max([0,col-N])
        Ux=np.min([len(new_image_sample),col+N])
        patch = new_image_sample[Lx:Ux]
        if len(patch) == N * 2:
            patches.append(entropy(patch))

    return(np.mean(patches))


def rearrange_feature_indices(mapping_filename, data_features):
    mapping = pd.read_csv(mapping_filename, header=None)
    mapping = mapping.values

    if np.min(mapping) < 0:
        mapping = mapping - np.min(mapping)

    mapping_dict = {}

    for i,j in enumerate(mapping):
        mapping_dict[j[0]] = i

    mapping_keys_sorted = sorted(mapping_dict.keys())

    oneD_mapping = []
    for i in mapping_keys_sorted:
        oneD_mapping.append(mapping_dict[i])

    dim = 300

    tsne_mapped_data_features = np.zeros((len(data_features), 300))


    for i,j in enumerate(tsne_mapped_data_features):
        for k,l in enumerate(oneD_mapping):
            tsne_mapped_data_features[i][k] = data_features[i][l]

    return(tsne_mapped_data_features)


def get_entropy_of_dataset(filename_prefix):
    input_filename = filename_prefix + ".csv"
    #output_filename = "test_quantized_original.csv"
    #output_filename = "test_quantized_randomized.csv"
    data = pd.read_csv(input_filename, header=None)
    data = data.head(1000)
    features= data.columns[1:]
    label = data.columns[0]
    data_features = data[features]
    data_label = data[label]
    data_features_np = data_features.to_numpy()
    data_label_np = data_label.to_numpy()

    entropy_dataset = []
    for i in range(len(data_features_np)):
        temp_new_sample = []
        entropy_dataset.append(get_entropy_sample_from_patch_entropy(data_features_np[i]))
        #temp_new_sample += find_nearest_patch_universal_dict(sample_patches)


    return(np.mean(entropy_dataset))
    #new_dataset_pd = pd.DataFrame(new_dataset)
    #new_dataset_pd.to_csv(output_filename, encoding='utf-8', index=False, header=None)

def get_entropy_of_datasets_for_diff_perplexity(filename_suffix, perplexity):

    data = pd.read_csv('test_randomized.csv', header=None)
    data = data.head(1000)
    features= data.columns[1:]
    label = data.columns[0]
    data_features = data[features]
    data_label = data[label]
    data_features_np = data_features.to_numpy()
    data_label_np = data_label.to_numpy()
    mapping_filename = "mapping" + filename_suffix + str(perplexity) + ".csv"

    tsne_mapped_dataset = rearrange_feature_indices(mapping_filename, data_features_np)
    entropy_dataset = []
    for i in range(len(tsne_mapped_dataset)):
        temp_new_sample = []
        entropy_dataset.append(get_entropy_sample_from_patch_entropy(tsne_mapped_dataset[i]))

    return(np.mean(entropy_dataset))

def main():
    original_entropy = get_entropy_of_dataset('test_transformed')
    randomized_entropy = get_entropy_of_dataset('test_randomized')

    perplexities = [10, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210, 230, 250, 270, 290, 310, 330, 350]

    temp_original = []
    temp_randomized = []
    temp_js = []
    temp_ecl = []
    for i in range(len(perplexities)):
        temp_original.append(original_entropy)
        temp_randomized.append(randomized_entropy)

    for perplexity in perplexities:
        temp_ecl.append(get_entropy_of_datasets_for_diff_perplexity('_ecl_', perplexity))
    for perplexity in perplexities:
        temp_js.append(get_entropy_of_datasets_for_diff_perplexity('_js_', perplexity))

    entropy_frame = []
    entropy_frame.append(perplexities)
    entropy_frame.append(temp_original)
    entropy_frame.append(temp_randomized)
    entropy_frame.append(temp_ecl)
    entropy_frame.append(temp_js)

    print(entropy_frame)
    entropy_frame_pd = pd.DataFrame(entropy_frame)
    entropy_frame_pd.to_csv('entropy_comparison_3.csv', encoding='utf-8', index=False, header=None)

if __name__ == "__main__":
    main()
