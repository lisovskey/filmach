from argparse import ArgumentParser
from glob import glob
from generate import sample_text
from dump import save_alphabet
from sys import stdout
from collections import Counter
from tensorflow import keras
from os import path
import numpy as np


def load_data(data_dir):
    """
    Load .txt files from `data_dir`.

    Split text into sequences of length `seq_len` and get chars
    after every sequence. Convert to one-hot arrays.

    Return features & labels.
    """
    global text, chars_indices, indices_chars

    texts = [open(filename).read()
             for filename in glob(path.join(data_dir, '*.txt'))]
    text = '\n\n'.join(texts)
    chars = sorted(list(set(text)))

    print(Counter(text))

    chars_indices = {char: i for i, char in enumerate(chars)}
    indices_chars = {i: char for i, char in enumerate(chars)}

    sequences = [text[i:i + seq_len]
                 for i in range(len(text) - seq_len)]
    next_chars = [text[i + seq_len]
                  for i in range(len(text) - seq_len)]

    x = np.zeros((len(sequences), seq_len, len(chars)), dtype=np.bool)
    y = np.zeros((len(sequences), len(chars)), dtype=np.bool)
    for i, sequence in enumerate(sequences):
        for j, char in enumerate(sequence):
            x[i, j, chars_indices[char]] = 1
        y[i, chars_indices[next_chars[i]]] = 1

    print('text length:', len(text))
    print('unique chars:', len(chars))
    print('total sequences:', len(sequences))

    return x, y


def demo_generation(epoch, logs):
    """
    Print demo generation on different diffusion while training.
    """
    print()
    print(5*'-', 'Generating text after epoch', epoch)
    start_index = np.random.randint(len(text) - seq_len)
    for diffusion in [0.2, 0.3, 0.4]:
        print(5*'-', 'Diffusion:', diffusion)
        sequence = text[start_index: start_index + seq_len]
        print(5*'-', f'Generating with seed: "{sequence}"')
        stdout.write(sequence)
        for char in sample_text(model, demo_length, chars_indices,
                                indices_chars, sequence, diffusion):
            stdout.write(char)
            stdout.flush()
        print()


def init_callbacks(model_path, tensorboard_dir):
    """
    Return `callbacks` with demo generator, model checkpoints
    and optional tensorboard.
    """
    callbacks = [
        keras.callbacks.LambdaCallback(on_epoch_end=demo_generation),
        keras.callbacks.ModelCheckpoint(model_path),
    ]
    if tensorboard_dir:
        callbacks.append(keras.callbacks.TensorBoard(tensorboard_dir,
                                                     write_images=True))
    return callbacks


def dense(x, layer_size, regularizer_rate):
    """
    Dense & batch norm layer
    """
    layer = keras.layers.Dense(
        units=layer_size,
        use_bias=False,
        kernel_regularizer=keras.regularizers.l2(regularizer_rate))
    x = layer(x)
    x = keras.layers.BatchNormalization()(x)
    return keras.layers.Activation('relu')(x)


def gru(x, layer_size, regularizer_rate):
    """
    Bidirectional GRU & batch norm layer
    """
    layer = keras.layers.Bidirectional(keras.layers.GRU(
        units=layer_size,
        activation=None,
        return_sequences=True,
        recurrent_regularizer=keras.regularizers.l2(regularizer_rate)))
    x = layer(x)
    x = keras.layers.BatchNormalization()(x)
    return keras.layers.Activation('tanh')(x)


def attention(x, layer_size):
    """
    Attention layer
    """
    x_attention = keras.layers.Dense(1, activation='tanh')(x)
    x_attention = keras.layers.Flatten()(x_attention)
    x_attention = keras.layers.Activation('softmax')(x_attention)
    x_attention = keras.layers.RepeatVector(layer_size)(x_attention)
    x_attention = keras.layers.Permute([2, 1])(x_attention)
    x = keras.layers.multiply([x, x_attention])
    return keras.layers.Lambda(lambda x: keras.backend.sum(x, axis=-2))(x)


def output_dense(x, layer_size):
    """
    Softmax dense layer.
    """
    return keras.layers.Dense(layer_size, activation='softmax')(x)


def init_model(input_shape, output_dim, recurrent_layers,
               layer_size, learning_rate, regularizer_rate):
    """
    Input: one hot sequence.

    Hidden: dense, GRUs, attention.

    Output: char index probas.
    """
    x_input = keras.layers.Input(shape=(input_shape))
    x = dense(x_input, layer_size, regularizer_rate)
    for _ in range(recurrent_layers):
        x = gru(keras.layers.concatenate([x_input, x]),
                layer_size, regularizer_rate)
    x = attention(x, keras.backend.shape(x)[-1])
    x_output = output_dense(x, output_dim)
    model = keras.Model(inputs=x_input, outputs=x_output)
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.RMSprop(learning_rate),
                  metrics=['accuracy'])
    print(model.summary())
    return model


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--epochs',
                        help='number of epochs to train',
                        type=int,
                        required=True)
    parser.add_argument('--seq_len',
                        help='length of sequences for text splitting',
                        type=int,
                        required=True)
    parser.add_argument('--batch_size',
                        help='minibatch size for training',
                        type=int,
                        default=128)
    parser.add_argument('--recurrent_layers',
                        help='number of gru layers',
                        type=int,
                        default=3)
    parser.add_argument('--layer_size',
                        help='length of recurrent layers',
                        type=int,
                        default=64)
    parser.add_argument('--learning_rate',
                        help='learning rate of optimizer',
                        type=float,
                        default=0.002)
    parser.add_argument('--validation_split',
                        help='part of training set for validation',
                        type=float,
                        default=0.1)
    parser.add_argument('--regularizer_rate',
                        help='recurrent dropout of recurrent layers',
                        type=float,
                        default=0.005)
    parser.add_argument('--data_dir',
                        help='directory with .txt files',
                        type=str,
                        default='data')
    parser.add_argument('--model_dir',
                        help='directory of model to save',
                        type=str,
                        default='checkpoints')
    parser.add_argument('--model_name',
                        help='name of model to save',
                        type=str,
                        default='model.h5')
    parser.add_argument('--alphabet_name',
                        help='name of alphabet to save',
                        type=str,
                        default='alphabet.pkl')
    parser.add_argument('--tensorboard_dir',
                        help='directory for tensorboard logs',
                        type=str,
                        default='')
    parser.add_argument('--demo_length',
                        help='minibatch size for training',
                        type=int,
                        default=300)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    seq_len, demo_length = args.seq_len, args.demo_length
    x, y = load_data(args.data_dir)
    save_alphabet(args.model_dir, args.alphabet_name,
                  chars_indices, indices_chars)
    model = init_model(x[0].shape, len(indices_chars),
                       args.recurrent_layers, args.layer_size,
                       args.learning_rate, args.regularizer_rate)
    callbacks = init_callbacks(path.join(args.model_dir,
                                         args.model_name),
                               args.tensorboard_dir)
    model.fit(x, y, batch_size=args.batch_size, epochs=args.epochs,
              callbacks=callbacks, validation_split=args.validation_split)
