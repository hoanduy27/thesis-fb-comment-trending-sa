import os
import numpy as np
import tensorflow as tf
import datetime
import copy
# from preprocess import read, tokenizer
from models.cnn import SALT, BSALT, SALTA
from models.losses import focal_loss, penalty_augmented_loss
from tensorflow.keras.optimizers import Adam, SGD, Adadelta

def get_SALT(**kwargs):
  config = {
    'model': 'salt', 
    'model_conf':
    {
      'input_length': kwargs['input_length'],
      'input_dim': kwargs['input_dim'],
      'embedding_dim': kwargs['embedding_dim'],
      'output_dim': kwargs['output_dim'],

      'num_kernels': kwargs.get('num_kernels', 3),
      'kernel_sizes': kwargs.get('kernel_sizes', 3),
      'pool_size': kwargs.get('pool_size', 2),
      'embedding_dropout': kwargs.get('embedding_dropout', 0.2),
      'conv_dropout': kwargs.get('conv_dropout', 0.2),
      'embedding_matrix': kwargs.get('embedding_matrix', None)
    }
  }

  salt_cv = SALT(
    **config['model_conf']
  )

  return config, salt_cv  

models = {
    'salt': get_SALT
}
class ModelCreator:
  """Utility for creating model
    Parameters:
  """
  
  def __init__(self, **kwargs):
    self.conf = kwargs
    self.LOSS_BASE_CONF = {
      'binary_crossentropy': ['binary_crossentropy', {}],
      'categorical_crossentropy': ['categorical_crossentropy', {}],
      'focal_loss': [focal_loss, {'alpha': 1, 'gamma': 2}],
      'penalty_augmented_loss': [penalty_augmented_loss, {'penalty_matrix': [], 'smooth': True}]
    }
    self.OPTIM_BASE_CONF = {
      'adam': [Adam, {'learning_rate': .001, 'beta_1': .9, 'beta_2': .999, 'epsilon': 1e-7, 'amsgrad': False}],
      'sgd': [SGD, {'learning_rate': 0.01, 'momentum': 0.0, 'nesterov': False}],
      'adadelta': [Adadelta, {'learning_rate': 0.001, 'rho': 0.95, 'epsilon': 1e-07}]
    }
    
  def __add_loss(self):
    output_dim = self.conf['model_conf']['output_dim']
    self.LOSS_BASE_CONF['penalty_augmented_loss'][1]['penalty_matrix'] = [[1]*output_dim]*output_dim
    
    loss_name = self.kwargs.get('loss', None)
    loss_conf = self.kwargs.get('loss_conf', {})

    if loss_name is None:
      if output_dim==1:
        loss_name = 'binary_crossentropy'
      else:
        loss_name = 'categorical_crossentropy'
    
    if loss_name in self.LOSS_BASE_CONF:
      # Get loss function and inital its configs
      loss, config = copy.deepcopy(self.LOSS_BASE_CONF[loss_name])
      # Update specified configs
      for param, val in loss_conf.items():
        if param in config:
          config[param] = val
        else:
          accepted_params = ', '.join(map(lambda pr: f'`{pr}`', config.keys()))
          raise Exception(f'Parameter `{param}` for `{loss_name}` loss is not defined. Accepted parameter: {accepted_params}')
      
      self.conf['loss'] = loss_name
      self.conf['loss_conf'] = config
      return loss if isinstance(loss, str) else loss(**config)
    else:
      accepted_losses = ', '.join(map(lambda pr: f'`{pr}`', LOSS_BASE_CONF.keys()))
      raise Exception(f'Loss is not supported. Supported losses: {accepted_losses}')

  def __add_optimizer(self):
    optim_name = self.kwargs.get('optim', 'adam')
    optim_conf = self.kwargs.get('optim_conf', {})
    
    if optim_name in self.OPTIM_BASE_CONF:
      optimizer, config = copy.deepcopy(self.OPTIM_BASE_CONF[optim_name])
      for param, val in optim_conf.items():
        if param in config:
          config[param] = val
        else:
          accepted_params = ', '.join(map(lambda pr: f'`{pr}`', config.keys()))
          raise Exception(f'Parameter `{param}` for `{optim}` optimizer is not defined. Accepted parameter: {accepted_params}')

      self.conf['optim'] = optim_name
      self.conf['optim_conf'] = config
    else:
      accepted_optims = ', '.join(map(lambda pr: f'`{pr}`', OPTIM_BASE_CONF.keys()))
      raise Exception(f'Optimizer is not supported. Supported optimizer: {accepted_optims}')

  def __get_optimizer(self):
    optimizer = self.OPTIM_BASE_CONF[self.conf['optim']][0]
    config = self.conf['optim_conf']
    return optimizer(**config)

  def __get_loss(self):
    loss = self.LOSS_BASE_CONF[self.conf['loss']][0]
    config = self.conf['loss_conf']
    return loss if isinstance(loss, str) else loss(**config)

  def build(self):
    model = self.conf['model']
    model_conf = self.conf['model_conf']
    model_conf, model = models[model](**model_conf)
    self.conf.update(model_conf)
    self.__add_loss()
    self.__add_optimizer()

    # Expected flow
    model.compile(
      loss = self.__get_loss(), 
      optimizer = self.__get_optimizer(),
      metrics = [tf.keras.metrics.AUC(),
                   'accuracy']
    )

    return self.conf, model


