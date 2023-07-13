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
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, ElectraForSequenceClassification


class ReviewData(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        """
        Initialize the ReviewData dataset.

        Args:
            encodings (dict): Input encodings.
            labels (list): Target labels.
        """
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        Args:
            idx (int): Index of the item.

        Returns:
            dict: Input encodings and target label.
        """
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.labels)


class EnsembleClassifier(nn.Module):
    def __init__(self):
        """
        Initialize the EnsembleClassifier model.
        """
        super(EnsembleClassifier, self).__init__()
        self.model1 = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
        self.model2 = ElectraForSequenceClassification.from_pretrained('google/electra-small-discriminator')
        self.dropout = nn.Dropout(0.3)
        self.out3 = nn.Linear(4, 2)

    def forward(self, ids):
        """
        Forward pass of the model.

        Args:
            ids (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # Pass through model 1
        hidden_state_1 = self.model1(ids, return_dict=False)
        dropout_output_1 = self.dropout(hidden_state_1[0])

        # Pass through model 2
        hidden_state_2 = self.model2(ids, return_dict=False)
        dropout_output_2 = self.dropout(hidden_state_2[0])

        # Concatenate the outputs
        concatenated_output = torch.cat((dropout_output_1, dropout_output_2), dim=1)

        # Pass through the final layer
        final_output = self.out3(concatenated_output)

        return final_output
