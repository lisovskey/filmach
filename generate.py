import sys
import random
import numpy as np


def sample(preds, diversity):
    """
    Take sample char from `preds` after softmax.
    """
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / diversity
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def sample_text(model, length, chars_indices, indices_chars, sequence,
                seq_len, diversity=1.0):
    """
    Yield predicted char after `sequence`.
    Shift `sequence` one char further `length` times.
    """
    for _ in range(length):
        x_pred = np.zeros((1, seq_len, len(chars_indices)))
        for i, char in enumerate(sequence):
            x_pred[0, i, chars_indices[char]] = 1.0
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indices_chars[next_index]
        sequence = sequence[1:] + next_char
        yield next_char
