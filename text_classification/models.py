"""Small recurrent networks built with the separately installed PyTorch library."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class SpamClassifier(nn.Module):
    """Learn embeddings, summarize the valid tokens, then predict ham/spam logits."""

    def __init__(self, vocabulary_size, embedding_size=50, hidden_size=80,
                 head_sizes=(10,), bidirectional=False):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_size, padding_idx=0)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True,
                            bidirectional=bidirectional)
        width = hidden_size * (2 if bidirectional else 1)
        layers = []
        for size in head_sizes:
            layers.extend([nn.Linear(width, size), nn.ReLU()])
            width = size
        self.head = nn.Sequential(*layers, nn.Linear(width, 2))

    def forward(self, tokens, lengths):
        if (tokens.ndim != 2 or lengths.ndim != 1 or len(lengths) != len(tokens)
                or (lengths < 1).any() or (lengths > tokens.shape[1]).any()):
            raise ValueError("Provide (batch, time) tokens and valid positive sequence lengths.")
        embedded = self.embedding(tokens)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        # Both final directions are needed: output[:, -1] is wrong for a padded BiLSTM.
        summary = torch.cat([hidden[-2], hidden[-1]], dim=1) if self.lstm.bidirectional else hidden[-1]
        return self.head(summary)  # CrossEntropyLoss consumes logits, without a softmax layer.


class SequenceModel(nn.Module):
    """Small shape demonstration: predict one output vector at every time step."""

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, features):
        sequence, _ = self.lstm(features)
        return self.output(sequence)
