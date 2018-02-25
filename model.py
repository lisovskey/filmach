from tensorflow import keras


def init_model(input_shape, output_dim, layer_size,
               learning_rate, dropout, recurrent_dropout):
    """
    Input: one hot sequence
    Hidden: 3 GRUs with batch normalization and dropout
    Output: char index
    """
    model = keras.models.Sequential()
    model.add(keras.layers.Bidirectional(
        keras.layers.GRU(units=layer_size,
                         dropout=dropout,
                         recurrent_dropout=recurrent_dropout,
                         return_sequences=True),
        input_shape=input_shape))
    model.add(keras.layers.Bidirectional(
        keras.layers.GRU(units=layer_size,
                         dropout=dropout,
                         recurrent_dropout=recurrent_dropout,
                         return_sequences=True)))
    model.add(keras.layers.Bidirectional(
        keras.layers.GRU(units=layer_size,
                         dropout=dropout, 
                         recurrent_dropout=recurrent_dropout,
                         return_sequences=False)))
    model.add(keras.layers.Dense(output_dim, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(lr=learning_rate),
                  metrics=['accuracy'])
    return model
