import pandas as pd
import numpy as np
from preprocess.normalize import normalize
from preprocess.augment import add_remove_accent_texts


def csv_to_dataset(filename, content_name, label_name='', preprocess=True, augmentations=[]):
  aug_dict = {'accent': add_remove_accent_texts}

  ds = pd.read_csv(filename)
  X = ds.iloc[:][content_name].values
  y = ds.iloc[:][label_name].values if label_name != '' else []
  if preprocess:
    X = np.vectorize(normalize)(X)
  for aug in augmentations:
    X, y = aug_dict[aug](X, y)
  
  return X.astype('object'), y
