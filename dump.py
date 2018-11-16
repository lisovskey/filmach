from os import path, mkdir
import pickle


def save_alphabet(directory, name, chars_indices, indices_chars):
    """
    Serialize `chars_indices` and `indices_chars`.
    """
    if not path.exists(directory):
        mkdir(directory)
    with open(path.join(directory, name) + '.pkl', 'wb') as file:
        pickle.dump((chars_indices, indices_chars), file)


def load_alphabet(directory, name):
    """
    Unserialize `chars_indices` and `indices_chars`.
    """
    with open(path.join(directory, name) + '.pkl', 'rb') as file:
        chars_indices, indices_chars = pickle.load(file)
    return chars_indices, indices_chars
