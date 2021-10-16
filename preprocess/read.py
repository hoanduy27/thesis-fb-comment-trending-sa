import pandas as pd
import numpy as np
from preprocess.normalize import normalize
from preprocess.augment import add_remove_accent_texts
from preprocess.tokenizer import WordSegmenter

def csv_to_dataset(filename, content_name, label_name='', preprocess=True, tokenize=None, augmentations=[]):
  aug_dict = {'accent': add_remove_accent_texts}

  ds = pd.read_csv(filename)
  X = ds.iloc[:][content_name].values
  y = ds.iloc[:][label_name].values if label_name != '' else []
  if preprocess:
    X = np.array(list(map(normalize, X)), dtype=object)
    # X = np.vectorize(normalize)(X)
  if tokenize is not None:
    X = np.array(list(map(WordSegmenter.tokenizer(tokenize), X)), dtype=object)
  for aug in augmentations:
    X, y = aug_dict[aug](X, y)
  
  return X.astype('object'), y