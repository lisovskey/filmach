import numpy as np


def sample(preds, candidates_num):
    """
    Take sample char from `preds` with `candidates_num` most
    possible char indices.
    """
    candidates = np.argpartition(preds, -candidates_num)
    return np.random.choice(candidates[-candidates_num:])


def sample_text(model, length, chars_indices, indices_chars, sequence,
                seq_len, candidates_num):
    """
    Yield predicted char after `sequence`.
    Shift `sequence` one char further `length` times.
    """
    x_pred = np.zeros((1, seq_len, len(chars_indices)))
    for i, char in enumerate(sequence):
        x_pred[0, i, chars_indices[char]] = 1.0
    for _ in range(length):
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, candidates_num)
        next_char = indices_chars[next_index]
        yield next_char
        x_pred = np.roll(x_pred, -1, axis=1)
        x_pred[0, -1] = np.zeros(len(chars_indices))
        x_pred[0, -1, chars_indices[next_char]] = 1.0
