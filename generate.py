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
    for _ in range(length):
        x_pred = np.zeros((1, seq_len, len(chars_indices)))
        for i, char in enumerate(sequence):
            x_pred[0, i, chars_indices[char]] = 1.0
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, candidates_num)
        next_char = indices_chars[next_index]
        sequence = sequence[1:] + next_char
        yield next_char
