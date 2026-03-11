import torch
import torch.nn as nn


class GRUSeq2Seq(nn.Module):
    def __init__(self,
                 input_size=28,
                 hidden_size=128,
                 num_layers=1,
                 output_size=10,
                 bidirectional=False,
                 dropout=0.0,
                 device="cpu"):

        super().__init__()

        self.device = device
        self.bidirectional = bidirectional

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )

        gru_out_size = hidden_size * (2 if bidirectional else 1)

        self.fc = nn.Linear(gru_out_size, output_size)

        self.to(device)

    def forward(self, x, lengths):
        """
        x: (batch, max_len, input_size)
        lengths: (batch,)
        """

        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        packed_out, _ = self.gru(packed)

        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out,
            batch_first=True
        )

        output = self.fc(output)  # (batch, max_len, output_size)

        return output