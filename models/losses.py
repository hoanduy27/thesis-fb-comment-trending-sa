from keras import backend as K
import tensorflow as tf

def penalty_augmented_loss(penalty_matrix, smooth=False):
  """Loss function using penalty matrix and categorical cross entropy, proposed in https://arxiv.org/abs/1806.08760"""
  penalty_matrix = K.constant(penalty_matrix, dtype=tf.float32)
  def calc(y_true, y_pred):

    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    
    true_class = tf.argmax(y_true, axis=1)
    pred_class = tf.argmax(y_pred, axis=1)

    # Smooth penalty: Column in penalty_matrix corresponding to the true class
    smooth_coeffs = tf.map_fn(lambda i: penalty_matrix[:, i], true_class, dtype=tf.float32)

    # Hard penalty: penalty_maxtrix(i, j), i in pred_class, j in true_class
    hard_weight = tf.map_fn(
        lambda i: penalty_matrix[pred_class[i], true_class[i]], 
        tf.range(len(true_class)), dtype=tf.float32
    )

    # Calculate penalty weight for cross entropy
    penalty_weight = K.sum(smooth_coeffs*y_pred, 1) if smooth else hard_weight

    # Return penalized cross entropy
    return penalty_weight * K.sum(-y_true * K.log(y_pred + K.epsilon()), 1)
  return calc

def focal_loss(alpha=.25, gamma=2.):
  """Focal loss as proposed in https://arxiv.org/abs/1708.02002"""
  def calc(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    return alpha*K.sum(-y_true * K.pow((1-y_pred), gamma) * tf.math.log(y_pred + K.epsilon()), 1)
  return calc