from argparse import ArgumentParser
from generate import sample_text
from tensorflow import keras


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--length',
                        help='length of generated text',
                        type=int)
    parser.add_argument('--model_name',
                        help='name of model to load',
                        type=str)
    parser.add_argument('--start_text',
                        help='first sequence to predict next',
                        type=str)
    parser.add_argument('--seq_len',
                        help='length of sequence and start text',
                        default=64,
                        type=int)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    error = 'sequence length and lenght of start text are not matching'
    assert args.seq_len == len(args.start_text), error
    model = keras.models.load_model(args.model_name)
    # TODO load chars_indices and indices_chars
    sample_text(model, args.length, chars_indices, indices_chars,
                args.start_text)
