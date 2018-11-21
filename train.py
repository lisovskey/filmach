from argparse import ArgumentParser
from glob import glob
from sample import sample_text
from dump import save_alphabet
from sys import stdout
from os import path
from model import Model
from tqdm import tqdm
import torch
import torch.utils.data
import numpy as np


def load_data(data_dir, sequence_size):
    """
    Load .txt files from `data_dir`.

    Split text into sequences of length `sequence_size` and get chars
    after every sequence. Convert to index numbers.

    Return `dataset`, `text`, `chars_indices` and `indices_chars`
    """
    texts = [open(filename).read()
             for filename in glob(path.join(data_dir, '*.txt'))]
    text = '\n'.join(texts)
    chars = sorted(list(set(text)))

    chars_indices = {char: i for i, char in enumerate(chars)}
    indices_chars = {i: char for i, char in enumerate(chars)}

    sequences = [[chars_indices[char] for char in text[i:i + sequence_size]]
                 for i in range(len(text) - sequence_size)]
    next_chars = [chars_indices[text[i + sequence_size]]
                  for i in range(len(text) - sequence_size)]

    x = torch.tensor(sequences, dtype=torch.long)
    y = torch.tensor(next_chars, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(x, y)

    print('text length:', len(text))
    print('unique chars:', len(chars))
    print('total sequences:', len(sequences))

    return dataset, text, chars_indices, indices_chars


def generate_demo(model, demo_length, text, sequence_size, chars_indices,
                  indices_chars):
    """
    Print demo generation on different diffusion after every epoch.
    """
    print()
    start_index = np.random.randint(len(text) - sequence_size)
    for diffusion in [0.2, 0.3, 0.4]:
        print(5*'-', 'Temperature:', diffusion)
        sequence = text[start_index: start_index + sequence_size]
        print(5*'-', f'Generating with seed: "{sequence}"')
        stdout.write(sequence)
        for char in sample_text(model, demo_length, chars_indices,
                                indices_chars, sequence, diffusion):
            stdout.write(char)
            stdout.flush()
        print()


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--sequence_size', type=int, required=True)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--layer_size', type=int, default=128)
    parser.add_argument('--embedding_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.005)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--model_dir', type=str, default='checkpoints')
    parser.add_argument('--model_name', type=str, default='model')
    parser.add_argument('--alphabet_name', type=str, default='alphabet')
    parser.add_argument('--demo_length', type=int, default=300)
    parser.add_argument('--disable_cuda', type=bool, default=False)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    dataset, text, chars_indices, indices_chars = load_data(args.data_dir,
                                                            args.sequence_size)
    save_alphabet(args.model_dir, args.alphabet_name,
                  chars_indices, indices_chars)
    model = Model(args.embedding_size, args.layer_size, args.sequence_size,
                  len(chars_indices), args.num_layers, args.dropout)
    if not args.disable_cuda and torch.cuda.is_available():
        model = model.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate,
                                    weight_decay=args.weight_decay)
    dataloader = torch.utils.data.DataLoader(dataset, args.batch_size,
                                             shuffle=True)

    for epoch in range(1, args.epochs + 1):
        progress = tqdm(dataloader)
        losses = []
        for x_batch, y_batch in progress:
            if next(model.parameters()).is_cuda:
                x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
            model.init_hidden(len(x_batch))
            y_preds = model(x_batch)
            loss = criterion(y_preds, y_batch)
            losses.append(loss.item())
            progress.set_description('Loss: {:.4f}'.format(np.mean(losses)))
            model.zero_grad()
            loss.backward()
            optimizer.step()
        generate_demo(model, args.demo_length, text, args.sequence_size,
                      chars_indices, indices_chars)
        model_name = f'{args.model_name}_{epoch}.pt'
        print(f'"{model_name}" saved')
        torch.save(model, path.join(args.model_dir, model_name))
