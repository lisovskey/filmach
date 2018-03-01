import pickle
import os


def save_alphabet(directory, filename, chars_indices, indices_chars):
    """
    Serialize `chars_indices` and `indices_chars`.
    """
    if not os.path.exists(directory):
        os.mkdir(directory)
    with open(os.path.join(directory, filename), 'wb') as file:
        pickle.dump((chars_indices, indices_chars), file)


def load_alphabet(directory, filename):
    """
    Unserialize `chars_indices` and `indices_chars`.
    """
    with open(os.path.join(directory, filename), 'rb') as file:
        chars_indices, indices_chars = pickle.load(file)
    return chars_indices, indices_chars
