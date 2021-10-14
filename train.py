from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard

import os
cur_dir = os.path.abspath(__file__)
cur_dir = os.path.dirname(cur_dir)

def create_callbacks(model_path, tensorboard_path, patience):
  save_model = ModelCheckpoint(model_path, save_best_only=True,monitor='val_accuracy', mode='max')
  reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1,  patience=patience, verbose=1, epsilon=1e-4, mode='min')
  tensorboard = TensorBoard(log_dir=tensorboard_path)
  return [mcp_save, reduce_lr_loss, tensorboard]

def train_SALT(name, X, y, n_folds=1, batch_size=8, epochs=10, **salt_kwargs, **checkpoint_kwargs):

  kfolds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=7).split(X, y)

  for i, (train_idx, val_idx) in enumerate(kfolds):
    salt_cv = SALT(
        input_length = kwargs.get('input_length'),
        input_dim = kwargs.get('input_dim'),
        embedding_dim = kwargs.get('embedding_dim'),
        output_dim = kwargs.get('output_dim'),
        num_kernels = kwargs.get('num_kernels'),
        kernel_size = kwargs.get('kernel_size'),
        pool_size = kwargs.get('pool_size'),
        embedding_dropout = kwargs.get('embedding_dropout'),
        conv_dropout = kwargs.get('conv_dropout')
        loss = kwargs.get('loss', None),
        embedding_matrix = kwargs.get('embedding_matrix', None)
    )
    callbacks = create_callbacks(
        model_path = f'{cur_dir}/logs/salt/{name}/fold_{i}/model',
        tensorboard_path = f'{cur_dir}/logs/salt/{name}/fold_{i}/tensorboard'),
        patience = kwargs.get('patience', 1)
    )
    salt_cv.fit(
      X[train_idx], 
      y[train_idx], 
      batch_size=batch_size,
      epochs=epochs, 
      validation_data=(test_seq, y_test), 
      callbacks = callbacks
    )

  


