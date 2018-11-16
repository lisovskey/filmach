from argparse import ArgumentParser
from generate import sample_text
from dump import load_alphabet
from sys import stdout
from os import path
from model import Model
import torch


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--length', type=int, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--start_text', type=str, required=True)
    parser.add_argument('--model_dir', type=str, default='checkpoints')
    parser.add_argument('--alphabet_name', type=str, default='alphabet')
    parser.add_argument('--diffusion', type=float, default=0.3)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    model = torch.load(path.join(args.model_dir, args.model_name) + '.pt')
    chars_indices, indices_chars = load_alphabet(args.model_dir,
                                                 args.alphabet_name)
    stdout.write(args.start_text)
    for char in sample_text(model, args.length, chars_indices,
                            indices_chars, args.start_text, args.diffusion):
        stdout.write(char)
        stdout.flush()
    print()
