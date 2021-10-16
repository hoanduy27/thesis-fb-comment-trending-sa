import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Flatten, Dense, Input, Softmax
from tensorflow.keras.layers import Conv1D, MaxPool1D, Dropout, BatchNormalization, LayerNormalization, MultiHeadAttention, Add
from tensorflow.keras.layers import concatenate
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import Adam
 
def MHSA(num_heads, key_dim, value_dim):
  from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention, Dense, Add
  def calc(input, attention_mask):
    mhsa = LayerNormalization()(input)
    mhsa = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, value_dim=value_dim, dropout=0.2)
    mhsa = mhsa(input, input, attention_mask=attention_mask)

    add1 = Add()([input, mhsa])
    add1 = LayerNormalization()(add1)

    return add1
  return calc

def SALT(input_length, input_dim, embedding_dim, output_dim, num_kernels, kernel_sizes, pool_size, embedding_dropout, conv_dropout, embedding_matrix=None):
    """SALT model, proposed in https://arxiv.org/abs/1806.08760"""
    inp = Input(shape=(input_length,))
    embedding = Embedding(input_dim=input_dim, output_dim=embedding_dim, input_length=input_length, mask_zero=False, weights=[embedding_matrix] if embedding_matrix is not None else None, name='embedding')(inp)
    embedding = Dropout(embedding_dropout)(embedding)

    conv = []

    if isinstance(kernel_sizes, int):
      kernel_sizes = [kernel_sizes]*num_kernels
    assert len(kernel_sizes) == num_kernels, 'If specify by list, the length of `kernel_sizes` must equal to `num_kernels`'
    for kernel_size in kernel_sizes:
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

    
    return model


def SALTA(input_length, input_dim, embedding_dim, output_dim, attention_heads, key_dim, value_dim, num_kernels, kernel_size, pool_size, embedding_dropout, conv_dropout, embedding_matrix=None, embedding_trainable=False):
    inp = Input(shape=(input_length,), name='input')
    inp_masked = Input(shape=(input_length,), name='input_masked')

    embedding = Embedding(input_dim=input_dim, output_dim=embedding_dim, input_length=input_length, mask_zero=False, weights=[embedding_matrix] if embedding_matrix is not None else None, trainable=embedding_trainable, name='embedding')(inp)
    embedding = Dropout(embedding_dropout)(embedding)

    mhsa = MHSA(attention_heads, key_dim, value_dim)(embedding, attention_mask=inp_masked)

    conv = []
    for i in range(num_kernels):
      conv += [Conv1D(1, kernel_size=kernel_size, padding='same', activation='relu', kernel_regularizer=l1_l2(0.01, 0.01))(mhsa)]
    
    concat = concatenate(conv)
    concat = BatchNormalization()(concat)

    conv1 = Conv1D(1, kernel_size=kernel_size, padding='same', activation='relu', kernel_regularizer=l1_l2(0.01, 0.01))(concat)
    conv1 = BatchNormalization()(conv1)

    add = Add()([concat, conv1])

    pool = MaxPool1D(pool_size=pool_size)(add)
    pool = BatchNormalization()(pool)

    flatten = Flatten()(pool)
    flatten = Dropout(rate=conv_dropout)(flatten)
    
    dense = Dense(64, activation='relu', kernel_regularizer=l1_l2(0.01, 0.01))(flatten)
    dense = Dense(output_dim, kernel_regularizer=l1_l2(0.01, 0.01))(dense)

    op = Softmax()(dense)

    model = Model(inputs = [inp, inp_masked], outputs = op)

    return model



def BSALT(input_length, input_dim, output_dim, num_kernels, kernel_size, pool_size, embedding_dropout, conv_dropout, bert_embedding):
    """Proposed: BSALT model"""
    inp = Input(shape=(input_length,), dtype='int32')
    inp_masked = Input(shape=(input_length,), dtype='int32')

    embedding = bert_embedding(inp, attention_mask=inp_masked)[0]
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

    model = Model(inputs = [inp, inp_masked], outputs = op)
    model.layers[2].trainable = False

    
    return model
