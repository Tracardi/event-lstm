import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Simulated data
data = ['A', 'B', 'C', 'D', 'E', 'x', 'A', 'C', 'x', 'B', 'D', 'x', 'E', 'A', 'B', 'x', 'C', 'D', 'E', 'B', 'A']
goal_event = 'x'

# Encode events to integers
event_to_ix = {event: i for i, event in enumerate(set(data))}
ix_to_event = {i: event for event, i in event_to_ix.items()}

# Parameters
sequence_length = 5
input_size = len(event_to_ix)
hidden_size = 10
num_layers = 1
output_size = 2  # Binary classification: leads to x or not

# Prepare sequences
sequences = []
labels = []
for i in range(len(data) - sequence_length):
    sequence = data[i:i+sequence_length]
    print(sequence)
    target = 1 if data[i+sequence_length] == goal_event else 0
    sequences.append(torch.tensor([event_to_ix[event] for event in sequence], dtype=torch.long))
    labels.append(target)

# One-hot encode sequences
sequences = torch.nn.functional.one_hot(torch.stack(sequences), num_classes=input_size).float()
labels = torch.tensor(labels)

# Create TensorDataset and DataLoader
dataset = TensorDataset(sequences, labels)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

class EventPredictor(nn.Module):
    def __init__(self):
        super(EventPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        x = self.fc(hn[-1])  # Take the last layer's hidden state
        return x

model = EventPredictor()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training loop
num_epochs = 350
for epoch in range(num_epochs):
    for sequences, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')


def predict_sequence(model, sequence):
    model.eval()
    with torch.no_grad():
        sequence_encoded = torch.tensor([event_to_ix[event] for event in sequence], dtype=torch.long)
        sequence_encoded = torch.nn.functional.one_hot(sequence_encoded.unsqueeze(0), num_classes=input_size).float()
        outputs = model(sequence_encoded)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_label = probabilities.argmax(1).item()
        return 'Leads to x' if predicted_label == 1 else 'Does not lead to x', probabilities[0][predicted_label].item()

# Example usage
test_sequence = ['A', 'B', 'C', 'D', 'E']
prediction, probability = predict_sequence(model, test_sequence)
print(f'Prediction: {prediction} with probability {probability:.4f}')
