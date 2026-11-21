
import torch
import torch.nn as nn


class CNN1DRegressor(nn.Module):

    def __init__(self, input_channels=17):

        super().__init__()

        # First Convolution Block
        self.conv1 = nn.Conv1d(
            in_channels=input_channels,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.relu1 = nn.ReLU()

        self.pool1 = nn.MaxPool1d(
            kernel_size=2,
            stride=2
        )

        # Second Convolution Block
        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.relu2 = nn.ReLU()

        self.pool2 = nn.MaxPool1d(
            kernel_size=2,
            stride=2
        )

        # Fully Connected Layer
        self.flatten = nn.Flatten()

        self.fc = nn.Linear(
            in_features=64 * 7,
            out_features=1
        )

    def forward(self, x):

        # Input: (batch, timesteps, features)
        # Conv1D expects: (batch, channels, length)
        x = x.transpose(1, 2)

        # First Convolution Block
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # Second Convolution Block
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Flatten and predict RUL
        x = self.flatten(x)
        x = self.fc(x)

        return x.squeeze(-1)


if __name__ == "__main__":

    model = CNN1DRegressor()

    # Create dummy input
    dummy_x = torch.randn(64, 30, 17)

    # Forward propagation
    predictions = model(dummy_x)

    print("Input shape:", dummy_x.shape)
    print("Output shape:", predictions.shape)

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print("Total parameters:", total_parameters)
