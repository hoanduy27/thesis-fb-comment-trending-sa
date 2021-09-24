import pandas as pd
import numpy as np
from preprocess.normalize import normalize
def csv_to_dataset(filename, content_name, label_name='', preprocess=True, augment=[]):
  ds = pd.read_csv(filename)
  X = ds.iloc[:][content_name].values
  y = ds.iloc[:][label_name].values if label_name != '' else []
  if preprocess:
    X = np.vectorize(normalize)(X)
  return X.astype('object'), y