from argparse import ArgumentParser
from generate import sample_text
from dump import load_alphabet
from tensorflow import keras
from sys import stdout
from os import path


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--length',
                        help='length of generated text',
                        type=int,
                        required=True)
    parser.add_argument('--start_text',
                        help='first sequence to predict next',
                        type=str,
                        required=True)
    parser.add_argument('--model_dir',
                        help='directory of model to load',
                        type=str,
                        default='checkpoints')
    parser.add_argument('--model_name',
                        help='name of model to load',
                        type=str,
                        default='model.h5')
    parser.add_argument('--alphabet_name',
                        help='name of alphabet to load',
                        type=str,
                        default='alphabet.pkl')
    parser.add_argument('--diffusion',
                        help='diffusion of sequences preds',
                        type=float,
                        default=0.3)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    model = keras.models.load_model(path.join(args.model_dir,
                                              args.model_name))
    chars_indices, indices_chars = load_alphabet(args.model_dir,
                                                 args.alphabet_name)
    stdout.write(args.start_text)
    for char in sample_text(model, args.length, chars_indices,
                            indices_chars, args.start_text, args.diffusion):
        stdout.write(char)
        stdout.flush()
    print()
