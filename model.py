from tensorflow import keras


def init_model(input_shape, output_dim, layer_size=128,
               learning_rate=0.001, dropout_rate=0.8,
               recurrent_dropout_rate=0.7):
    model = keras.models.Sequential()
    model.add(keras.layers.Bidirectional(
        keras.layers.GRU(units=layer_size,
                         dropout=dropout_rate,
                         recurrent_dropout=recurrent_dropout_rate,
                         return_sequences=True),
        input_shape=input_shape))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Bidirectional(
        keras.layers.GRU(units=layer_size,
                         dropout=dropout_rate, 
                         recurrent_dropout=recurrent_dropout_rate,
                         return_sequences=False)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(output_dim, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(lr=learning_rate),
                  metrics=['accuracy'])
    return model
