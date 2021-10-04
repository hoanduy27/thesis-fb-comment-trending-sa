import numpy as np
from models.utils import remove_accents

def add_remove_accent_texts(texts, labels=[]):
    """Augment dataset by appending texts that have accents removed"""
    texts_aug = np.vectorize(remove_accents)(texts)
    texts_aug = np.append(texts, texts_aug, 0)
    if len(label) != 0:
        labels_aug = np.append(labels, labels, 0)
        return texts_aug, labels_aug
    else:
        return texts_aug, []
