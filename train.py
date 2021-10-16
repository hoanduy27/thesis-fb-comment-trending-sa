import os
import numpy as np
import tensorflow as tf
import datetime
import copy
# from preprocess import read, tokenizer
from models.cnn import SALT, BSALT, SALTA
from models.losses import focal_loss, penalty_augmented_loss

from sklearn.model_selection import StratifiedKFold
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard, CSVLogger
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, SGD, Adadelta

cur_dir = os.path.abspath(__file__)
cur_dir = os.path.dirname(cur_dir)


def create_callbacks(model_path, tensorboard_path, log_path, factor, cooldown, patience):
  save_model = ModelCheckpoint(model_path, save_best_only=True,monitor='val_accuracy', mode='max')
  reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', mode='min', factor=factor,  cooldown=cooldown, patience=patience, min_delta=1e-4, verbose=1)
  tensorboard = TensorBoard(log_dir=tensorboard_path)
  log_train = CSVLogger(log_path, separator=",", append=False)
  return [save_model, reduce_lr_loss, tensorboard, log_train]

def fake_train(**kwargs):
  optim = kwargs.get('optim', 'adam')
  optim_conf = kwargs.get('optim_conf', {})
  
  return get_optimizer(optim, optim_conf)

def preprocess_data(train_data):
  # Load dataset
  print("Loading dataset...")

  X,y = read.csv_to_dataset(train_data, 'comment', 'label', preprocess=False, tokenize=None)

  # Segmentation
  print("Segmenting...")

  X = np.array(list(map(tokenizer.WordSegmenter.segmentize('vncorenlp'), X)), dtype=object)

  # Tokenization
  print("Tokenizing...")

  phobert_dict = f'{cur_dir}/../lexicon'
  class BPE():
    bpe_codes = f'{phobert_dir}/bpe.codes'
  
  bpe = fastBPE(BPE())

  vocab = Dictionary()
  vocab.add_from_file(f'{phobert_dir}/dict.txt')

  maxlen, vocab_size, X_seq, X_masked = tokenizer.text_2_bpe_sequences(X, bpe, vocab, maxlength='auto', truncating=True, padding=True)

  # TODO: augmentation

  return X_seq, X_masked, maxlen, vocab_size


def train_model(X, X_masked, y, n_folds=1, batch_size=8, epochs=10,):

    # Save model config
    if not os.path.exists(model_path):
      os.makedirs(model_path)
    tf.keras.utils.plot_model(model, to_file=f'{model_path}/model.png', show_shapes=True, show_dtype=True)
    from contextlib import redirect_stdout
    with open(f'{model_path}/summary.txt', 'w') as f:
      with redirect_stdout(f):
        model.summary()

    # Callbacks
    callbacks = create_callbacks(
      model_path = f'{model_dir}/fold_{i}/model',
      tensorboard_path = f'{model_dir}/fold_{i}/tensorboard', 
      log_path = f'{model_dir}/fold_{i}/history.csv',
      factor = kwargs.get('factor', .1),
      cooldown= kwargs.get('cooldown', 2),
      patience = kwargs.get('patience', 1)
    )

    # Start training
    salt_cv.fit(
      X[train_idx], 
      y_cat[train_idx], 
      batch_size=batch_size,
      epochs=epochs, 
      validation_data=(X[val_idx], y_cat[val_idx]), 
      callbacks = callbacks
    )
    print()



def train_BSALT(name, X, X_masked, y, n_folds=1, batch_size=8, epochs=10, **kwargs):
  model_dir = f'{cur_dir}/logs/bsalt/{name}'
  kfolds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=7).split(X, y)
  y_cat = to_categorical(y)
  
  optim = kwargs.get('optim', 'adam')
  optim_conf = kwargs.get('optim_conf', {})

  for i, (train_idx, val_idx) in enumerate(kfolds):
    print(f'Fold {i}:')
    print('>'*60)

    bsalt_cv = BSALT(
      input_length = kwargs.get('input_length'),
      input_dim = kwargs.get('input_dim'),
      output_dim = kwargs.get('output_dim', y_cat.shape[1]),
      num_kernels = kwargs.get('num_kernels', 3),
      kernel_size = kwargs.get('kernel_size', 3),
      pool_size = kwargs.get('pool_size', 2),
      embedding_dropout = kwargs.get('embedding_dropout', 0.2),
      conv_dropout = kwargs.get('conv_dropout', 0.2),
      bert_embedding = kwargs.get('bert_embedding')
    )
    # Save model config
    if i==0:
      if not os.path.exists(model_dir):
        os.makedirs(model_dir)

      tf.keras.utils.plot_model(bsalt_cv, to_file=f'{model_dir}/model.png', show_shapes=True, show_dtype=True)

      from contextlib import redirect_stdout
      with open(f'{model_dir}/summary.txt', 'w') as f:
        with redirect_stdout(f):
          bsalt_cv.summary()

      
    # Callbacks
    callbacks = create_callbacks(
      model_path = f'{model_dir}/fold_{i}/model',
      tensorboard_path = f'{model_dir}/fold_{i}/tensorboard', 
      log_path = f'{model_dir}/fold_{i}/history.csv',
      factor = kwargs.get('factor', .1),
      cooldown= kwargs.get('cooldown', 2),
      patience = kwargs.get('patience', 1)
    )
    
    # Loss
    loss = kwargs.get('loss', None)
    if loss is None:
      if output_dim==1:
        loss = 'binary_crossentropy'
      else:
        loss = 'categorical_crossentropy'
        
    # Compile model
    optimizer = get_optimizer(optim, optim_conf)
    bsalt_cv.compile(
        loss = loss, 
        optimizer = optimizer,
        metrics = [tf.keras.metrics.AUC(),
                   'accuracy']
    )

    # Start training
    bsalt_cv.fit(
      [X[train_idx], X_masked[train_idx]],
      y_cat[train_idx], 
      batch_size=batch_size,
      epochs=epochs, 
      validation_data=([X[val_idx], X_masked[val_idx]], y_cat[val_idx]), 
      callbacks = callbacks
    )
    print()

def train_SALTA(name, X, X_masked, y, n_folds=1, batch_size=8, epochs=10, **kwargs):
  model_dir = f'{cur_dir}/logs/salta/{name}'
  kfolds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=7).split(X, y)
  y_cat = to_categorical(y)
  
  optim = kwargs.get('optim', 'adam')
  optim_conf = kwargs.get('optim_conf', {})

  for i, (train_idx, val_idx) in enumerate(kfolds):
    print(f'Fold {i}:')
    print('>'*60)

    salta_cv = SALTA(
      input_length = kwargs.get('input_length'),
      input_dim = kwargs.get('input_dim'),
      embedding_dim = kwargs.get('embedding_dim'),
      output_dim = kwargs.get('output_dim', y_cat.shape[1]),
      attention_heads = kwargs.get('attention_heads', 4), 
      key_dim = kwargs.get('key_dim', 256), 
      value_dim = kwargs.get('value_dim', 256),
      num_kernels = kwargs.get('num_kernels', 3),
      kernel_size = kwargs.get('kernel_size', 3),
      pool_size = kwargs.get('pool_size', 2),
      embedding_dropout = kwargs.get('embedding_dropout', 0.2),
      conv_dropout = kwargs.get('conv_dropout', 0.2),
      embedding_matrix = kwargs.get('embedding_matrix', None),
      embedding_trainable = kwargs.get('embedding_trainable', False)
    )
    # Save model config
    if i==0:
      if not os.path.exists(model_dir):
        os.makedirs(model_dir)

      tf.keras.utils.plot_model(salta_cv, to_file=f'{model_dir}/model.png', show_shapes=True, show_dtype=True)

      from contextlib import redirect_stdout
      with open(f'{model_dir}/summary.txt', 'w') as f:
        with redirect_stdout(f):
          salta_cv.summary()

      
    # Callbacks
    callbacks = create_callbacks(
      model_path = f'{model_dir}/fold_{i}/model',
      tensorboard_path = f'{model_dir}/fold_{i}/tensorboard', 
      log_path = f'{model_dir}/fold_{i}/history.csv',
      factor = kwargs.get('factor', .1),
      cooldown= kwargs.get('cooldown', 2),
      patience = kwargs.get('patience', 1)
    )
    
    # Loss
    loss = kwargs.get('loss', None)
    if loss is None:
      if output_dim==1:
        loss = 'binary_crossentropy'
      else:
        loss = 'categorical_crossentropy'
        
    # Compile model
    optimizer = get_optimizer(optim, optim_conf)
    salta_cv.compile(
        loss = loss, 
        optimizer = optimizer,
        metrics = [tf.keras.metrics.AUC(),
                   'accuracy']
    )

    # Start training
    salta_cv.fit(
      [X[train_idx], X_masked[train_idx]],
      y_cat[train_idx], 
      batch_size=batch_size,
      epochs=epochs, 
      validation_data=([X[val_idx], X_masked[val_idx]], y_cat[val_idx]), 
      callbacks = callbacks
    )
    print()


get_model = {
  'salt': get_SALT 
}