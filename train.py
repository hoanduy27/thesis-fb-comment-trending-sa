import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard, CSVLogger
from tensorflow.keras.utils import to_categorical
from models.cnn import SALT


cur_dir = os.path.abspath(__file__)
cur_dir = os.path.dirname(cur_dir)

def create_callbacks(model_path, tensorboard_path, log_path, patience):
  save_model = ModelCheckpoint(model_path, save_best_only=True,monitor='val_accuracy', mode='max')
  reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1,  patience=patience, verbose=1, epsilon=1e-4, mode='min')
  tensorboard = TensorBoard(log_dir=tensorboard_path)
  log_train = CSVLogger(log_path, separator=",", append=False)
  return [save_model, reduce_lr_loss, tensorboard, log_train]

def train_SALT(name, X, y, n_folds=1, batch_size=8, epochs=10, **kwargs):
  model_dir = f'{cur_dir}/logs/salt/{name}'
  kfolds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=7).split(X, y)
  y_cat = to_categorical(y)
  
  for i, (train_idx, val_idx) in enumerate(kfolds):
    print(f'Fold {i}:')
    print('>'*60)

    salt_cv = SALT(
      input_length = kwargs.get('input_length'),
      input_dim = kwargs.get('input_dim'),
      embedding_dim = kwargs.get('embedding_dim'),
      output_dim = kwargs.get('output_dim', y_cat.shape[1]),
      num_kernels = kwargs.get('num_kernels', 3),
      kernel_size = kwargs.get('kernel_size', 3),
      pool_size = kwargs.get('pool_size', 2),
      embedding_dropout = kwargs.get('embedding_dropout', 0.2),
      conv_dropout = kwargs.get('conv_dropout', 0.2),
      loss = kwargs.get('loss', None),
      embedding_matrix = kwargs.get('embedding_matrix', None)
    )
    # Save model config
    if i==0:
      if not os.path.exists(model_dir):
        os.makedirs(model_dir)
      tf.keras.utils.plot_model(salt_cv, to_file=f'{model_dir}/model.png', show_shapes=True, show_dtype=True)
      from contextlib import redirect_stdout
      with open(f'{model_dir}/summary.txt', 'w') as f:
        with redirect_stdout(f):
          salt_cv.summary()

      
    # Callbacks
    callbacks = create_callbacks(
      model_path = f'{model_dir}/fold_{i}/model',
      tensorboard_path = f'{model_dir}/fold_{i}/tensorboard', 
      log_path = f'{model_dir}/fold_{i}/history.csv',
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
