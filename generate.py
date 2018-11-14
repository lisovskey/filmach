import numpy as np


def sample_text(model, length, chars_indices, indices_chars, sequence,
                diffusion):
    """
    Yield predicted char after `sequence`.
    Shift `sequence` one char further `length` times.
    """
    def sample(preds, diffusion):
        """
        Make a diffusion of chars predictions and take most possible.
        """
        preds = [pred + np.random.uniform(high=diffusion) for pred in preds]
        return np.argmax(preds)

    x_pred = np.zeros([1, len(sequence)])
    for i, char in enumerate(sequence):
        x_pred[0, i] = chars_indices[char]
    for _ in range(length):
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, diffusion)
        next_char = indices_chars[next_index]
        yield next_char
        x_pred = np.roll(x_pred, -1, axis=1)
        x_pred[0, -1] = chars_indices[next_char]
