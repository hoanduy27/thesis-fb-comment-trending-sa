import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Flatten, Dense, Input, Softmax
from tensorflow.keras.layers import Conv1D, MaxPool1D, Dropout, BatchNormalization
from tensorflow.keras.layers import concatenate
from tensorflow.keras.regularizers import l1_l2
 
 
def SALT(input_length, input_dim, embedding_dim, output_dim, num_kernels, kernel_size, pool_size, embedding_dropout, conv_dropout, loss=None, embedding_matrix=None):
    """SALT model, proposed in https://arxiv.org/abs/1806.08760"""
    inp = Input(shape=(input_length,))
    embedding = Embedding(input_dim=input_dim, output_dim=embedding_dim, input_length=input_length, mask_zero=False, weights=[embedding_matrix] if embedding_matrix is not None else None)(inp)
    embedding = Dropout(embedding_dropout)(embedding)

    conv = []
    for i in range(num_kernels):
      conv += [Conv1D(1, kernel_size=kernel_size, padding='same', activation='relu', kernel_regularizer=l1_l2(0.01, 0.01))(embedding)]
    
    concat = concatenate(conv)
    concat = BatchNormalization()(concat)

    pool = MaxPool1D(pool_size=pool_size)(concat)
    pool = BatchNormalization()(pool)

    flatten = Flatten()(pool)
    flatten = Dropout(rate=conv_dropout)(flatten)
    
    dense = Dense(64, activation='relu', kernel_regularizer=l1_l2(0.01, 0.01))(flatten)
    dense = Dense(output_dim, kernel_regularizer=l1_l2(0.01, 0.01))(dense)

    op = Softmax()(dense)

    model = Model(inputs = inp, outputs = op)

    if loss is not None:
        loss = loss
    else:
      if output_dim==1:
        loss = 'binary_crossentropy'
      else:
        loss = 'categorical_crossentropy'
        
    model.compile(
        loss = loss, 
        optimizer = Adam(learning_rate=.001, beta_1=.9, beta_2=.99, epsilon=1e-8),
        metrics = [tf.keras.metrics.AUC(),
                   'accuracy']
    )
    return model