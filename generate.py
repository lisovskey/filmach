import numpy as np
import torch


def sample_text(model, length, chars_indices, indices_chars, sequence,
                diffusion, use_cuda=False):
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
        x_pred = torch.tensor(x_pred, dtype=torch.long)
        if use_cuda:
            x_pred = x_pred.cuda()
        preds = model(x_pred, use_cuda)[0]
        next_index = sample(preds, diffusion)
        next_char = indices_chars[next_index]
        yield next_char
        x_pred = np.roll(x_pred, -1, axis=1)
        x_pred[0, -1] = chars_indices[next_char]
