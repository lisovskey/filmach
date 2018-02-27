from argparse import ArgumentParser
from glob import glob
from model import init_model
from generate import sample_text
from tensorflow import keras
from time import time
import os
import pickle
import numpy as np
import sys
import random


def load_data(data_dir, sequence_len, step=2):
    """
    Load .txt files from `data_dir`.

    Split text into sequences of length `seq_len` with offset `step`
    and get chars after every sequence.
    Convert to one hot arrays.

    Return `x`, `y`
    """
    global seq_len, text, chars_indices, indices_chars

    texts = [open(filename).read()
             for filename in glob(os.path.join(data_dir, '*.txt'))]
    text = '\n'.join(texts)
    chars = sorted(list(set(text)))

    chars_indices = {char: i for i, char in enumerate(chars)}
    indices_chars = {i: char for i, char in enumerate(chars)}

    seq_len = sequence_len
    sequences = [text[i:i + seq_len]
                 for i in range(0, len(text) - seq_len, step)]
    next_chars = [text[i + seq_len]
                  for i in range(0, len(text) - seq_len, step)]

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


def save_alphabet(model_dir, alphabet_name):
    """
    serialize `chars_indices` and `indices_chars` to `model_dir` folder
    with `alphabet_name` filename.
    """
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    with open(os.path.join(model_dir, alphabet_name), 'wb') as file:
        pickle.dump((chars_indices, indices_chars), file)


def demo_generation(epoch, logs):
    """
    print demo generation on different diversities while training.
    """
    print()
    print(5*'-', 'Generating text after epoch', epoch)
    start_index = random.randint(0, len(text) - seq_len - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print(5*'-', 'Diversity:', diversity)
        sequence = text[start_index: start_index + seq_len]
        print(5*'-', f'Generating with seed: "{sequence}"')
        sys.stdout.write(sequence)
        for char in sample_text(model, 256, chars_indices,
                                indices_chars, sequence,
                                seq_len, diversity):
            sys.stdout.write(char)
            sys.stdout.flush()
        print()


def init_callbacks(model_path, tensorboard_dir=None):
    """
    Return `callbacks` with demo generator, model checkpoints
    and optional tensorboard.
    """
    callbacks = [
        keras.callbacks.LambdaCallback(on_epoch_end=demo_generation),
        keras.callbacks.ModelCheckpoint(model_path),
    ]
    if tensorboard_dir:
        callbacks.append(keras.callbacks.TensorBoard(
            tensorboard_dir, write_images=True))
    return callbacks


def train_model(model, x, y, epochs, batch_size, callbacks):
    """
    Fit model and note training time.
    Return `model`.
    """
    start = time()
    model.fit(x, y, batch_size=batch_size, epochs=epochs,
              callbacks=callbacks)
    print(f'training took: {time() - start // 60} minutes')
    return model


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--epochs',
                        help='number of epochs to train',
                        type=int,
                        required=True)
    parser.add_argument('--batch_size',
                        help='minibatch size for training',
                        type=int,
                        required=True)
    parser.add_argument('--seq_len',
                        help='length of sequences',
                        type=int,
                        required=True)
    parser.add_argument('--layer_size',
                        help='length of recurrent layers',
                        type=int,
                        default=128)
    parser.add_argument('--learning_rate',
                        help='learning rate of optimizer',
                        type=float,
                        default=0.001)
    parser.add_argument('--dropout',
                        help='dropout of recurrent layers',
                        type=float,
                        default=0.75)
    parser.add_argument('--recurrent_dropout',
                        help='recurrent dropout of recurrent layers',
                        type=float,
                        default=0.5)
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
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    x, y = load_data(args.data_dir, args.seq_len)
    save_alphabet(args.model_dir, args.alphabet_name)
    model = init_model(x[0].shape, len(indices_chars), args.layer_size,
                       args.learning_rate, args.dropout,
                       args.recurrent_dropout)
    callbacks = init_callbacks(os.path.join(args.model_dir,
                                            args.model_name),
                               args.tensorboard_dir)
    model = train_model(model, x, y, args.epochs, args.batch_size,
                        callbacks)
