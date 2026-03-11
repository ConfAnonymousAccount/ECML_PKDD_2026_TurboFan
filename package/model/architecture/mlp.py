import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self,
                 input_size=28,
                 hidden_sizes=(128, 64),
                 output_size=10,
                 activation=nn.ReLU,
                 dropout=0.0,
                 device="cpu"):

        super().__init__()
        self.name = "MLP"
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation = activation
        self.dropout = dropout
        self.device = device

        self._build_model()
        self.to(self.device)

    def _build_model(self):

        layers = []
        in_features = self.input_size

        # Build hidden layers dynamically
        for hidden_size in self.hidden_sizes:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(self.activation())

            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))

            in_features = hidden_size  # update input for next layer

        # Output layer
        layers.append(nn.Linear(in_features, self.output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)