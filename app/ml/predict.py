import torch
import os
from app.ml.model import get_model
from app.utils.serialization import load_pickle

_path = os.path.dirname(os.path.abspath(__file__))


# Training loop
def predict_next_events(sequence, num_predictions):
    event_to_ix = load_pickle(os.path.join(_path, 'data/event_to_ix.pickle'))
    ix_to_event = load_pickle(os.path.join(_path, 'data/ix_to_event.pickle'))

    # Parameters
    input_size = len(event_to_ix)  # number of unique events

    model = get_model(input_size)
    model.load_state_dict(torch.load(os.path.join(_path, 'data/event_lstm_model.pythorch')))
    model.eval()

    predictions = []
    probabilities = []
    input_sequence = torch.tensor([event_to_ix[event] for event in sequence], dtype=torch.long)
    input_sequence = torch.nn.functional.one_hot(input_sequence.unsqueeze(0), num_classes=input_size).float()

    with torch.no_grad():
        for _ in range(num_predictions):
            logits = model(input_sequence)
            probability = torch.softmax(logits, dim=1)
            predicted_idx = probability.argmax(1).item()
            predicted_event = ix_to_event[predicted_idx]
            predictions.append(predicted_event)
            probabilities.append(probability[0][predicted_idx].item())
            # Update the input_sequence to include the new prediction
            new_event_tensor = torch.nn.functional.one_hot(torch.tensor([predicted_idx]),
                                                           num_classes=input_size).float()
            input_sequence = torch.cat((input_sequence[:, 1:, :], new_event_tensor.unsqueeze(0)), dim=1)

    return predictions, probabilities
