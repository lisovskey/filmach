from argparse import ArgumentParser
from glob import glob
from generate import sample_text
from dump import save_alphabet
from sys import stdout
from tensorflow import keras
from os import path
import numpy as np


def load_data(data_dir):
    """
    Load .txt files from `data_dir`.

    Split text into sequences of length `seq_len` and get chars
    after every sequence. Convert to index numbers.

    Return features & labels.
    """
    global text, chars_indices, indices_chars

    texts = [open(filename).read()
             for filename in glob(path.join(data_dir, '*.txt'))]
    text = '\n'.join(texts)
    chars = sorted(list(set(text)))

    chars_indices = {char: i for i, char in enumerate(chars)}
    indices_chars = {i: char for i, char in enumerate(chars)}

    sequences = [[chars_indices[char] for char in text[i:i + seq_len]]
                 for i in range(len(text) - seq_len)]
    next_chars = [chars_indices[text[i + seq_len]]
                  for i in range(len(text) - seq_len)]

    x = np.array(sequences)
    y = np.array(next_chars)

    print('text length:', len(text))
    print('unique chars:', len(chars))
    print('total sequences:', len(sequences))

    return x, y


def demo_generation(epoch, logs):
    """
    Print demo generation on different diffusion after every `epoch`.
    """
    print()
    print(5*'-', 'Generating text after epoch', epoch + 1)
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


def embedding(x, input_dim, output_dim):
    """
    Embedding layer.
    """
    layer = keras.layers.Embedding(input_dim, output_dim)
    return layer(x)


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


def output_dense(x, layer_size):
    """
    Softmax dense layer.
    """
    x = keras.layers.Flatten()(x)
    layer = keras.layers.Dense(layer_size, activation='softmax')
    return layer(x)


def init_model(input_shape, output_dim, recurrent_layers, layer_size,
               embedding_size, learning_rate, regularizer_rate):
    """
    Input: one hot sequence.

    Hidden: dense, GRUs, attention.

    Output: char index probas.
    """
    x_input = keras.layers.Input(shape=input_shape)
    x_embedded = embedding(x_input, output_dim, embedding_size)
    x_gru = x_embedded
    for _ in range(recurrent_layers):
        x_embedded = dense(x_embedded, layer_size, regularizer_rate)
        x_gru = gru(keras.layers.concatenate([x_embedded, x_gru]),
                    layer_size, regularizer_rate)
    x_output = output_dense(x_gru, output_dim)
    model = keras.Model(inputs=x_input, outputs=x_output)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=keras.optimizers.Nadam(learning_rate),
                  metrics=['accuracy'])
    model.summary()
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
    parser.add_argument('--embedding_size',
                        help='length of embedding layer',
                        type=int,
                        default=16)
    parser.add_argument('--learning_rate',
                        help='learning rate of optimizer',
                        type=float,
                        default=0.001)
    parser.add_argument('--regularizer_rate',
                        help='recurrent and kernel regularizers',
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
                        help='demonstration text length',
                        type=int,
                        default=300)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    seq_len, demo_length = args.seq_len, args.demo_length
    x, y = load_data(args.data_dir)
    save_alphabet(args.model_dir, args.alphabet_name,
                  chars_indices, indices_chars)
    model = init_model(x[0].shape, len(indices_chars), args.recurrent_layers,
                       args.layer_size, args.embedding_size,
                       args.learning_rate, args.regularizer_rate)
    callbacks = init_callbacks(path.join(args.model_dir,
                                         args.model_name),
                               args.tensorboard_dir)
    model.fit(x, y, batch_size=args.batch_size, epochs=args.epochs,
              callbacks=callbacks)
