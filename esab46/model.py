import torch
from torch.nn.utils import rnn as torch_rnn


class JenNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # TODO(locbui): consider dropout & bidirectional
        self.gru = torch.nn.GRU(input_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths):
        packed_x = torch_rnn.pack_padded_sequence(x, lengths, enforce_sorted=False)
        packed_o, hidden = self.gru(packed_x)
        output, _ = torch_rnn.pad_packed_sequence(packed_o)
        squeezed_hidden = hidden.squeeze(0)
        pred = self.fc(squeezed_hidden)
        return pred
