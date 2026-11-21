import torch
import torch.nn as nn


class LSTMRegressor(nn.Module):

    def __init__(
        self,
        input_size=17,
        hidden_size=64
    ):
        super().__init__()

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        # Output layer
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):

        # Process the input sequence
        _, (hidden_state, cell_state) = self.lstm(x)

        # Get the final hidden state
        last_hidden = hidden_state[-1]

        # Predict RUL
        prediction = self.fc(last_hidden)

        return prediction.squeeze(-1)


if __name__ == "__main__":

    model = LSTMRegressor()

    # Create a dummy batch
    dummy_x = torch.randn(64, 30, 17)

    # Forward propagation
    predictions = model(dummy_x)

    print("Input shape:", dummy_x.shape)
    print("Output shape:", predictions.shape)
    print("Total parameters:",
          sum(p.numel() for p in model.parameters()))