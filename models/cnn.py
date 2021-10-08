def SALT(input_length, input_dim, embedding_dim, output_dim, num_kernels, kernel_size, pool_size, embedding_dropout, conv_dropout,penalty=None, use_penalty=False, penalty_smooth=False, embedding_matrix=[]):
    inp = Input(shape=(input_length,))
    embedding = Embedding(input_dim=input_dim, output_dim=embedding_dim, input_length=input_length, mask_zero=False)(inp)
    dropout1 = Dropout(embedding_dropout)(embedding)
    conv = []
    for i in range(num_kernels):
      conv += [Conv1D(1, kernel_size=kernel_size, padding='same', activation='relu', kernel_regularizer=l1_l2(0.01, 0.01))(dropout1)]

    
    concat = concatenate(conv)
    concat = BatchNormalization()(concat)

    pool = MaxPool1D(pool_size=pool_size)(concat)
    pool = BatchNormalization()(pool)

    flatten = Flatten()(pool)
    dropout2 = Dropout(rate=conv_dropout)(flatten)
    
    dense = Dense(64, activation='relu', kernel_regularizer=l1_l2(0.01, 0.01))(dropout2)
    dense = Dense(output_dim, kernel_regularizer=l1_l2(0.01, 0.01))(dense)

    op = Softmax()(dense)

    model = Model(inputs = inp, outputs = op)

    if not use_penalty:
        if output_dim==2:
            loss = 'binary_crossentropy'
        else:
            loss = 'categorical_crossentropy'
    else:
        loss = penalty_augmented_loss(penalty, smooth=penalty_smooth)

    model.compile(
        loss = loss, 
        optimizer = 'adam', 
        metrics = [tf.keras.metrics.AUC(),
                   'accuracy']
    )
    return model
