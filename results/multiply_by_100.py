import pandas as pd
import numpy as np

filename_prefix = "rf_ecl_har1d"
filename = filename_prefix + ".csv"
output_filename = filename_prefix + "_100.csv"


data = pd.read_csv(filename)
data = data.values
data = data * 100
data_100 = pd.DataFrame(data)
data_100.to_csv(output_filename, encoding='utf-8', index=False, header=None)
