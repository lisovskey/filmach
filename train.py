from argparse import ArgumentParser
from glob import glob
from model import init_model
from generate import sample_text
from tensorflow import keras
import numpy as np
import sys
import random


seq_len: int
text: str
chars_indices: dict
indices_chars: dict


def load_data(data_dir, seq_len=64, step=4):
    """
    Load .txt files from `data_dir`.

    Split text into sequences of length `seq_len` with offset `step`
    and get chars after every sequence.
    Convert to one hot arrays.
    
    Return `x`, `y`, `chars_indices`, `indices_chars`
    """
    texts = [open(filename).read()
             for filename in glob(f'{data_dir}/*.txt')]
    text = '\n'.join(texts)
    chars = sorted(list(set(text)))

    chars_indices = {char: i for i, char in enumerate(chars)}
    indices_chars = {i: char for i, char in enumerate(chars)}

    sequences = [text[i: i + seq_len]
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
    print('total chars:', len(chars))
    print('total sequences:', len(sequences))

    return x, y, chars_indices, indices_chars


def demo_generation(epoch, logs):
    print()
    print(5*'-', 'Generating text after epoch', epoch)
    start_index = random.randint(0, len(text) - seq_len - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print(5*'-', 'Diversity:', diversity)
        sentence = text[start_index: start_index + seq_len]
        print(5*'-', f'Generating with seed: "{sentence}"')
        sys.stdout.write(sentence)
        sample_text(model, 256, chars_indices, indices_chars, sentence,
                    seq_len, diversity)
        print()


def init_callbacks(tensorboard_dir=None, model_name=None):
    callbacks = [
        keras.callbacks.LambdaCallback(on_epoch_end=demo_generation),
        keras.callbacks.ModelCheckpoint(model_name),
    ]
    if tensorboard_dir:
        callbacks.append(keras.callbacks.TensorBoard(tensorboard_dir))
    return callbacks


def train_model(model, x, y, epochs, batch_size, callbacks):
    model.fit(x, y, batch_size=batch_size, epochs=epochs,
              callbacks=init_callbacks())
    return model


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_dir',
                        help='directory with .txt files',
                        type=str)
    parser.add_argument('--epochs',
                        help='number of epochs to train',
                        type=int)
    parser.add_argument('--batch_size',
                        help='minibatch size for training',
                        type=int)
    parser.add_argument('--model_name',
                        help='name of model to save',
                        type=str,
                        default='model.h5')
    parser.add_argument('--tensorboard_dir',
                        help='directory for tensorboard logs',
                        type=str,
                        default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    x, y, chars_indices, indices_chars = load_data(args.data_dir)
    model = init_model(x[0].shape, len(indices_chars))
    callbacks = init_callbacks(tensorboard_dir=args.tensorboard_dir,
                               model_name=args.model_name)
    model = train_model(model, x, y, args.epochs, args.batch_size,
                        init_callbacks(args.tensorboard_dir))
