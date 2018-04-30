from os import path, mkdir
import pickle


def save_alphabet(directory, filename, chars_indices, indices_chars):
    """
    Serialize `chars_indices` and `indices_chars`.
    """
    if not path.exists(directory):
        mkdir(directory)
    with open(path.join(directory, filename), 'wb') as file:
        pickle.dump((chars_indices, indices_chars), file)


def load_alphabet(directory, filename):
    """
    Unserialize `chars_indices` and `indices_chars`.
    """
    with open(path.join(directory, filename), 'rb') as file:
        chars_indices, indices_chars = pickle.load(file)
    return chars_indices, indices_chars
