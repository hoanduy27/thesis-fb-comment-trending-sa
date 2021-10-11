import numpy as np
from models.utils import remove_accents

def add_remove_accent_texts(texts, labels=[]):
    """Augment dataset by appending texts that have accents removed"""
    if type(texts[0]) == str:
      texts_aug = np.array(list(map(remove_accents, texts)), dtype=object)
    else:
      texts_aug = np.array(list(list(map(remove_accents, text)) for text in texts), dtype='object')
    texts_aug = np.append(texts, texts_aug, 0)
    if len(labels) != 0:
        labels_aug = np.append(labels, labels, 0)
        return texts_aug, labels_aug
    else:
        return texts_aug, []

