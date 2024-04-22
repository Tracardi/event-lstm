import torch
import os
from torch.utils.data import DataLoader, TensorDataset

from app.ml.model import get_model, get_criterion, get_optimizer
from app.utils.serialization import save_pickle

_path = os.path.dirname(os.path.abspath(__file__))

# Data: event sequences and the next event
data = ['A', 'B', 'C', 'A', 'C', 'B', 'A', 'B']


def train_model(data, num_epochs):
    # Encode events as numeric values
    event_to_ix = {event: i for i, event in enumerate(set(data))}
    ix_to_event = {i: event for event, i in event_to_ix.items()}

    # Parameters
    input_size = len(event_to_ix)  # number of unique events

    model = get_model(input_size)
    optimizer = get_optimizer(model)
    criterion = get_criterion()

    # Prepare data
    sequences = [torch.tensor([event_to_ix[event] for event in data[i:i + 3]], dtype=torch.long) for i in
                 range(len(data) - 3)]
    print(sequences)
    next_events = [torch.tensor(event_to_ix[data[i + 3]], dtype=torch.long) for i in range(len(data) - 3)]
    dataset = TensorDataset(
        torch.nn.functional.one_hot(
            torch.stack(sequences),
            num_classes=input_size).float(),
        torch.stack(next_events))
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Training loop
    for epoch in range(num_epochs):
        for sequence, target in dataloader:
            optimizer.zero_grad()
            output = model(sequence)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch}, Loss: {loss.item()}')

    # Save the model
    model_path = os.path.join(_path, 'data/event_lstm_model.pythorch')
    torch.save(model.state_dict(), model_path)
    print("Model saved to", model_path)

    save_pickle(os.path.join(_path, 'data/event_to_ix.pickle'), event_to_ix)
    save_pickle(os.path.join(_path, 'data/ix_to_event.pickle'), ix_to_event)
    print("Saved Indexes.")
