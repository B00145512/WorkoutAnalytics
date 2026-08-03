import torch
import numpy as np

def predict_rep_lstm(sequence, model, mean, std):
    sequence = (sequence - mean) / std
    
    tensor = torch.tensor(sequence, dtype=torch.float32)

    # ensure shape is (151, 25)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)   # → (1, 151, 25)

# if already 3D, do nothing

    model.eval()
    with torch.no_grad():
        output = model(tensor)
        pred = torch.argmax(output, dim=1).item()

    return pred

def fix_sequence_length(seq):
    if len(seq) > 151:
        return seq[:151]
    elif len(seq) < 151:
        last = seq[-1]
        pad = [last] * (151 - len(seq))
        return np.vstack([seq, pad])
    return seq