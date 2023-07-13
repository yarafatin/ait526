"""
# AIT 526 - Natural Language Processing
# Final Project - Predicting Yelp Rating Polarity
# Group 9:
# Yasser Parambathkandy
# Indranil Pal
# 7/12/2023
"""

import torch
import torch.nn as nn

# Global variables
use_packed_sequence = True  # Whether to use packed sequences for variable length input
embedding_dim = 50  # Dimensionality of word embeddings
hidden_size = 16  # Size of hidden state in LSTM
n_layers = 3  # Number of LSTM layers
lstm_drop = 0.5  # Dropout rate for LSTM layers
lin_drop = 0.5  # Dropout rate for linear layers


class LSTMModule(nn.Module):
    """
    LSTMModule is a module that consists of an LSTM network for sequence classification.
    It takes variable-length sequences as input and produces output predictions.

    Args:
        vocab_size (int): The size of the vocabulary.
        seq_len (int): The maximum sequence length.
        hidden_size (int, optional): The size of the hidden state in the LSTM. Defaults to 16.
        n_layers (int, optional): The number of LSTM layers. Defaults to 3.
    """

    def __init__(self, vocab_size, seq_len, hidden_size=hidden_size, n_layers=n_layers):
        super(LSTMModule, self).__init__()
        self.seq_len = seq_len
        # Embedding layer for converting word indices to dense word vectors
        self.embedding = nn.Embedding(vocab_size + 2, embedding_dim, padding_idx=0)
        # LSTM layer
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=n_layers, dropout=lstm_drop)
        # Linear layer for computing the mean over the time dimension
        self.mean = nn.Linear(seq_len * hidden_size, hidden_size)
        # Batch normalization layer for stabilizing the training process
        self.bn_mean = nn.BatchNorm1d(hidden_size)
        # Output linear layer
        self.out = nn.Linear(hidden_size, 2)
        # Dropout layer for regularizing the output
        self.drop = nn.Dropout(lin_drop)

    def forward(self, x):
        # Embedding layer
        x = self.embedding(x).permute(1, 0, 2)
        if use_packed_sequence:
            # Compute sequence lengths for packed sequences
            lengths = []
            for sentence_idx in range(x.shape[1]):
                n_zeros = torch.sum(x[:, sentence_idx, :] == 0) / 50
                lengths.append(self.seq_len - n_zeros.item())
            # Pack the sequences based on their lengths
            x = nn.utils.rnn.pack_padded_sequence(x, lengths, enforce_sorted=False)
        # LSTM layer
        lstm_out, _ = self.lstm(x)

        if use_packed_sequence:
            # Pad the packed sequences back to the original length
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, total_length=self.seq_len)

        # Compute the mean over the time dimension
        mean_over_t = self.drop(self.bn_mean(self.mean(lstm_out.permute(1, 0, 2).reshape(lstm_out.shape[1], -1))))

        # Output linear layer
        return self.out(mean_over_t)
