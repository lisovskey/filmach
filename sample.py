from argparse import ArgumentParser
from generate import sample_text
from dump import load_alphabet
from tensorflow import keras
import sys
import os


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
    parser.add_argument('--seq_len',
                        help='length of sequence and start text',
                        type=int,
                        required=True)
    parser.add_argument('--candidates_num',
                        help='number of candidates for next char',
                        type=int,
                        default=3)
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
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    assert args.seq_len == len(args.start_text), \
            'sequence length and length of start text are not equal'
    model = keras.models.load_model(os.path.join(args.model_dir,
                                                 args.model_name))
    chars_indices, indices_chars = load_alphabet(args.model_dir,
                                                 args.alphabet_name)
    for char in sample_text(model, args.length, chars_indices,
                            indices_chars, args.start_text,
                            args.seq_len, args.candidates_num):
        sys.stdout.write(char)
        sys.stdout.flush()
