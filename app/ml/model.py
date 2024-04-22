import torch.nn as nn
import torch.optim as optim


def get_model(input_size, num_layers=1, hidden_size=10):
    output_size = input_size  # output size is the same as input because we're predicting the same events

    class EventLSTM(nn.Module):
        def __init__(self):
            super(EventLSTM, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            output, (hn, cn) = self.lstm(x)
            output = self.fc(output[:, -1, :])  # Get the output of the last time step
            return output

    # Initialize model, loss function, and optimizer

    return EventLSTM()


def get_optimizer(model):
    return optim.Adam(model.parameters(), lr=0.01)


def get_criterion():
    return nn.CrossEntropyLoss()
