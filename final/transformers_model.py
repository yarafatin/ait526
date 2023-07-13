"""
# AIT 526 - Natural Language Processing
# Final Project - Predicting Yelp Rating Polarity
# Group 9:
# Yasser Parambathkandy
# Indranil Pal
# 7/12/2023
"""
import os

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import DistilBertForSequenceClassification, ElectraForSequenceClassification, AdamW
from transformers import DistilBertTokenizerFast, get_scheduler

import data_load
from transformer_class import ReviewData, EnsembleClassifier

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"device used {device}")

# Tuning parameters
num_epochs = 1  # Number of training epochs
max_len = 512  # Maximum length of input sequences
model_distilbert = 'distilbert'  # Name of the DistilBERT model
model_electra = 'electra'  # Name of the Electra model
model_ensemble = 'ensemble'  # Name of the ensemble model
learning_rate = 0.00005  # Learning rate for optimizer during training


def process():
    """
    Process the training and evaluation steps for multiple models.
    """
    train_dataloader, val_dataloader, test_dataloader = load_datasets()
    print('loaded datasets')

    model_names = [model_distilbert, model_electra, model_ensemble]

    print('start training')
    for model_name in model_names:
        num_training_steps = num_epochs * len(train_dataloader)
        model = train_model(model_name, train_dataloader, num_training_steps)
        predictions, true_labels = evaluate_model(model, model_name, val_dataloader)

        # Compute evaluation metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions)
        recall = recall_score(true_labels, predictions)
        print(f"{model_name} model - validation accuracy:{accuracy}, precision:{precision}, recall:{recall}")

        save_model(model, model_name)

        test_model(model_name, test_dataloader)
    print('completed training')


def load_datasets():
    """
    Load and preprocess the train and validation datasets.

    Returns:
        train_dataset (ReviewData): Preprocessed training dataset.
        val_dataset (ReviewData): Preprocessed validation dataset.
    """
    # Load data
    train_df, val_df, test_df = data_load.load_data()
    # Extract texts and labels from train, test, and validation data
    train_texts, train_labels = train_df["text"].tolist(), train_df["label"].tolist()
    val_texts, val_labels = val_df["text"].tolist(), val_df["label"].tolist()
    test_texts, test_labels = test_df["text"].tolist(), test_df["label"].tolist()
    # Initialize tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained('google/electra-small-discriminator')
    # Tokenize train, test, and validation texts
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_len)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=max_len)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=max_len)

    train_dataloader = DataLoader(ReviewData(train_encodings, train_labels), shuffle=True, batch_size=6)
    val_dataloader = DataLoader(ReviewData(val_encodings, val_labels), batch_size=6)
    test_dataloader = DataLoader(ReviewData(test_encodings, test_labels), batch_size=6)

    return train_dataloader, val_dataloader, test_dataloader


def train_model(model_name, train_dataloader, num_training_steps):
    """
    Train a specified model on the training dataset.

    Args:
        model_name (str): Name of the model.
        train_dataloader (DataLoader): DataLoader for the training dataset.
        num_training_steps (int): Number of total training steps.

    Returns:
        model: Trained model.
    """
    print(f"Start training model based on {model_name}")

    # Load the appropriate model
    model = get_model_by_name(model_name)
    model.to(device)

    # Initialize optimizer, scheduler, and progress bar
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0,
                                 num_training_steps=num_training_steps)
    progress_bar = tqdm(range(num_training_steps))

    # Model training
    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            if (model_name == model_distilbert) or (model_name == model_electra):
                outputs = model(input_ids, labels=labels)
                loss = outputs[0]
            else:
                outputs = model(input_ids)
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            progress_bar.update(1)

    return model


def get_model_by_name(model_name):
    """
    Get the model based on the specified name.

    Args:
        model_name (str): Name of the model.

    Returns:
        torch.nn.Module: Model instance.
    """
    if model_name == model_distilbert:
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
    elif model_name == model_electra:
        model = ElectraForSequenceClassification.from_pretrained('google/electra-small-discriminator')
    else:
        model = EnsembleClassifier()
    return model


def evaluate_model(model, model_name, eval_dataloader):
    """
    Evaluate the specified model on the evaluation dataset.

    Args:
        model (torch.nn.Module): Model to be evaluated.
        model_name (str): Name of the model.
        eval_dataloader (DataLoader): DataLoader for the evaluation dataset.

    Returns:
        tuple: Predictions and true labels.
    """
    predictions = []
    true_labels = []

    num_eval_steps = len(eval_dataloader)
    progress_bar = tqdm(range(num_eval_steps))

    # Model evaluation
    model.eval()
    with torch.no_grad():
        for batch in eval_dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            if (model_name == model_distilbert) or (model_name == model_electra):
                outputs = model(input_ids, labels=labels)
                logits = outputs.logits
            else:
                outputs = model(input_ids)
                logits = outputs

            # Get predicted classes and true labels
            predicted_classes = torch.argmax(logits, dim=-1)
            predicted_classes = predicted_classes.detach().cpu().numpy()
            true_labels_batch = labels.detach().cpu().numpy()

            # Append to the overall predictions and true labels
            predictions.extend(predicted_classes)
            true_labels.extend(true_labels_batch)

            progress_bar.update(1)

    return predictions, true_labels


def save_model(model, model_name):
    """
    Save the trained model.

    Args:
        model (torch.nn.Module): Trained model.
        model_name (str): Name of the model.
    """
    model_path = f"model_{model_name}.pt"

    if os.path.exists(model_path):
        os.remove(model_path)

    # Save the new model
    torch.save(model.state_dict(), model_path)
    print(f"model '{model_path}' saved")


def test_model(model_name, test_dataloader):
    """
    Test the specified model on the test dataset.

    Args:
        model_name (str): Name of the model.
        test_dataloader (DataLoader): DataLoader for the test dataset.
    """
    # Get the model based on the model name
    model = get_model_by_name(model_name)

    # Load the model state dict
    model.load_state_dict(torch.load(f"model_{model_name}.pt", map_location=device))
    model.to(device)

    # Evaluate the model on the test dataset
    predictions, true_labels = evaluate_model(model, model_name, test_dataloader)

    # Compute evaluation metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)

    # Print the evaluation metrics
    print(f"{model_name} model - Test data accuracy: {accuracy}, precision: {precision}, recall: {recall}")


if __name__ == '__main__':
    process()
