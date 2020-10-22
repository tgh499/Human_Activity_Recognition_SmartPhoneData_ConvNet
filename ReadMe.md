![Accuracy for different perplexities of t-SNE. Also shows how our feature embedding method achieved superior resutls using Euclidean and JS distance metrics.](HARUS_accuracy.png)

**Figure**: Accuracy for different perplexities of t-SNE using the transformed Human Activitiy Recognition Using Smartphones (HARUS) dataset. Also shows how our feature embedding method achieved superior results using Euclidean and JS distance metrics.


![HARUS Entropy](HARUS_entropy.png)

**Figure**: Entropy after applying the feature embedding method to the same datasets is much lower. Because of the specific way entropy was calculated, this experiment proves the introduction of local structures in the samples.


**Directions**:
Please go to https://dl.acm.org/doi/10.1145/3299815.3314429 for details of the feature embedding method. Once done, use the files in the following order.

How it works?

1. get a non-image dataset that satisfies the contraints of the above paper
2. generate distance_matrix using js_geodesic.py
3. generate t-SNE mappings for Jensen-Shannon (JS) using the distance matrix, and for Euclidean distance use the training set;
4. find the array representation using generate_tsne_mapped_dataset_ecl.py or generate_tsne_mapped_dataset_js.py
5. use the datasets with CNN; I used PyTorch; You may use TensorFlow. It doesn't matter.

If the performance doesn't improve, use Hungarian method. 

To measure entropy, use the quantize_dataset_singlefile_bucket.py file. Note that entropy can be measured in many ways. Here we have tried to divide the entire range of feature values into buckets, and use the bucket index as a patch label.
