import numpy as np
import torch


def sample_text(model, length, chars_indices, indices_chars, sequence,
                temperature):
    """
    Yield predicted char after `sequence`.
    Shift `sequence` one char further `length` times.
    """
    model.init_hidden(1)
    weight = next(model.parameters())
    x_pred = [[chars_indices[char] for char in sequence]]
    x_pred = weight.new_tensor(x_pred, dtype=torch.long)
    with torch.no_grad():
        for _ in range(length):
            preds = model(x_pred).squeeze()
            char_weights = preds.div(temperature).exp()
            next_index = torch.multinomial(char_weights, 1).squeeze().item()
            yield indices_chars[next_index]
            x_pred = torch.cat([x_pred[:, 1:], x_pred[:, :1]], dim=1)
            x_pred[0, -1] = next_index
