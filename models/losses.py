import tensorflow as tf

def penalty_augmented_loss(penalty_matrix, smooth=False):
  penalty_matrix = tf.constant(penalty_matrix, dtype=tf.float32)
  def calc(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    
    # Dot product on each batch
    true_class = tf.argmax(y_true, axis=1)
    pred_class = tf.argmax(y_pred, axis=1)

    # Column in penalty_matrix corresponding to the true class
    smooth_coeffs = tf.map_fn(lambda i: penalty_matrix[:, i], true_class, dtype=tf.float32)
    # Position in penalty matrix correspoding to [pred_class, true_class]
    hard_weight = tf.map_fn(
        lambda i: penalty_matrix[pred_class[i], true_class[i]], 
        tf.range(len(true_class)), dtype=tf.float32
    )
    penalty_weight = tf.math.reduce_sum(smooth_coeffs*y_pred, 1) if smooth else hard_weight
    
    cross_ent = penalty_weight * tf.math.reduce_sum(-y_true * tf.math.log(0.01+y_pred), 1)
    # for i in range(len(y_true)):
    #   dot_p = -tf.tensordot(y_true[i], tf.math.log(y_pred[i]), 1)

    #   true_class = tf.argmax(y_true[i])
    #   pred_class = tf.argmax(y_pred[i])

    #   penalty_weight = tf.tensordot(penalty_matrix[:, true_class], y_pred[i], 1) if smooth\
    #     else penalty_matrix[pred_class, true_class]

    #   cross_ent = tf.concat([cross_ent, [dot_p]], 0)

    return cross_ent
  return calc