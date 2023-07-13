"""
# AIT 526 - Natural Language Processing
# Final Project - Predicting Yelp Rating Polarity
# Group 9:
# Yasser Parambathkandy
# Indranil Pal
# 7/12/2023
"""

import nltk
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm

import data_load
from lstm_class import LSTMModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Device to be used for computation (GPU or CPU)
random_seed = 42  # Random seed for reproducibility
torch.manual_seed(random_seed)  # Set random seed for PyTorch
np.random.seed(random_seed)  # Set random seed for NumPy
model_save_file_name = "model_lstm.pt"  # File name to save the trained LSTM model
glove_embedding_file = "glove.6B.50d.txt"  # File path for pre-trained GloVe word embeddings

# Training parameters to tune
embedding_dim = 50  # Dimensionality of word embeddings
n_epochs = 1  # Number of training epochs
learning_rate = 0.001  # Learning rate for optimizer
batch_size = 32  # Number of sequences per batch during training
hidden_size = 16  # Size of hidden state in LSTM
num_layers = 3  # Number of LSTM layers
lstm_dropout = 0.5  # Dropout rate for LSTM layers
linear_dropout = 0.5  # Dropout rate for linear layers


def train_model(model, optimizer, criterion, x_train, y_train):
    """
    Trains a model using the provided optimizer and criterion.

    Args:
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        criterion: The loss criterion used for training.
        x_train (torch.Tensor): The input training data.
        y_train (torch.Tensor): The target training data.
        batch_size (int): The batch size used during training. Default is 32.

    Returns:
        float: The average training loss per step.
    """

    model.train()
    num_samples = len(x_train)
    num_batches = num_samples // batch_size
    train_loss = 0.0
    train_steps = 0

    with tqdm(total=num_batches, desc="Training") as progress_bar:
        for batch in range(num_batches + 1):
            # Calculate the start and end indices for the current batch
            start_index = batch * batch_size
            end_index = min((batch + 1) * batch_size, num_samples)
            indices = slice(start_index, end_index)
            # Zero the gradients
            optimizer.zero_grad()
            # Retrieve the current batch
            batch_x = x_train[indices]
            batch_y = y_train[indices]
            # Forward pass
            logits = model(batch_x)
            # Calculate the loss
            loss = criterion(logits, batch_y)
            # Backward pass
            loss.backward()
            # Update the model parameters
            optimizer.step()
            # Update the training loss and steps
            train_loss += loss.item()
            train_steps += 1
            # Update the progress bar
            progress_bar.update(1)
    return train_loss / train_steps


def calculate_accuracy(predictions, targets, model):
    """
    Calculates the accuracy of model predictions given the targets.

    Args:
        predictions (torch.Tensor): The predicted labels.
        targets (torch.Tensor): The target labels.
        model (torch.nn.Module): The model used for predictions.

    Returns:
        float: The accuracy score.
    """
    with torch.no_grad():
        num_samples = len(predictions)
        num_batches = (num_samples + batch_size - 1) // batch_size
        logits = torch.empty(num_samples, 2)

        for batch in range(num_batches):
            # Calculate the start and end indices for the current batch
            start_index = batch * batch_size
            end_index = min((batch + 1) * batch_size, num_samples)
            indices = slice(start_index, end_index)
            # Retrieve the current batch predictions
            batch_predictions = predictions[indices]
            # Forward pass and store the logits
            logits[indices] = model(batch_predictions)
        # Convert logits to numpy array and calculate predicted labels
        predicted_labels = np.argmax(logits.cpu().numpy(), axis=1)

    # Calculate and return the accuracy score
    return 100 * accuracy_score(targets.cpu().numpy(), predicted_labels)


def evaluate_model(model, x, y):
    """
    Evaluates a model using the provided inputs and targets.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        x (torch.Tensor): The input data.
        y (torch.Tensor): The target data.

    Returns:
        tuple: A tuple containing the accuracy score and predicted labels.
    """
    model.eval()
    with torch.no_grad():
        num_samples = len(x)
        logits = torch.empty(num_samples, 2)

        for batch in range((num_samples + batch_size - 1) // batch_size):
            # Calculate the start and end indices for the current batch
            start_index = batch * batch_size
            end_index = min((batch + 1) * batch_size, num_samples)
            indices = slice(start_index, end_index)
            # Retrieve the current batch inputs
            batch_inputs = x[indices]
            # Forward pass and store the logits
            logits[indices] = model(batch_inputs)
        # Convert logits to numpy array and calculate predicted labels
        predicted_labels = np.argmax(logits.cpu().numpy(), axis=1)
    # Calculate the accuracy score
    accuracy = 100 * accuracy_score(y.cpu().numpy(), predicted_labels)

    # Return the accuracy score and predicted labels
    return accuracy, predicted_labels


def test_model(model, x_test, y_test):
    """
    Tests a model using the provided test data and prints the accuracy and confusion matrix.

    Args:
        model (torch.nn.Module): The model to be tested.
        x_test (torch.Tensor): The test input data.
        y_test (torch.Tensor): The test target data.
        model_save_file_name (str): The file name to load the model's state dictionary from.

    Returns:
        None
    """
    model.load_state_dict(torch.load(model_save_file_name))
    # Evaluate the model
    accuracy, predicted_labels = evaluate_model(model, x_test, y_test)
    # Print the accuracy
    print(f"Test data accuracy: {accuracy}")
    # Print the confusion matrix
    print("Confusion matrix:")
    print(confusion_matrix(y_test.cpu().numpy(), predicted_labels))


def tokenize_reviews(train_reviews, test_reviews, validation_reviews):
    """
    Tokenizes the input sentences and creates a vocabulary dictionary.

    Args:
        train_reviews (np.array): The training sentences.
        test_reviews (np.array): The development sentences.
        validation_reviews (np.array): The validation sentences.

    Returns:
        tuple: A tuple containing the token vocabulary dictionary and the maximum sequence length.
    """
    # Combine all reviews for tokenization and sequence length calculation
    all_reviews = np.concatenate((train_reviews, test_reviews, validation_reviews))
    tokens = []
    max_seq_len = 0
    for review in all_reviews:
        # Tokenize the sentence
        tokens_in_review = nltk.word_tokenize(review)
        # Extend the tokens list with tokens from the current sentence
        tokens.extend(tokens_in_review)
        # Update the maximum sequence length if needed
        max_seq_len = max(max_seq_len, len(tokens_in_review))

    # Create the token vocabulary dictionary
    token_vocab = {key: i for i, key in enumerate(set(tokens), start=1)}
    return token_vocab, max_seq_len


def load_word_embeddings(review_dict):
    """
    Loads GloVe word embeddings and creates a dictionary of embeddings for the words in the vocabulary.

    Args:
        review_dict (dict): The vocabulary dictionary containing word-to-index mapping.

    Returns:
        dict: A dictionary containing word embeddings for words in the vocabulary.
    """
    embeddings_dict = {}

    with open(glove_embedding_file, "r") as file:
        embedding_data = file.read()

    for line in embedding_data.split("\n")[:-1]:
        text = line.split()
        if text[0] in review_dict:
            # Convert the embedding values to a tensor
            embedding = torch.from_numpy(np.array(text[1:], dtype="float32"))
            # Store the embedding in the dictionary using the word index as the key
            embeddings_dict[review_dict[text[0]]] = embedding
    return embeddings_dict


def reviews_to_tokenid_map(reviews, review_dict, max_length):
    """
    Converts input sentences to a matrix of token IDs.

    Args:
        reviews (list): The input sentences.
        review_dict (dict): The vocabulary dictionary containing word-to-ID mapping.
        max_length (int): The maximum length of the sequences.

    Returns:
        np.array: A 2D NumPy array representing the token IDs for the sentences.
    """
    num_sentences = len(reviews)
    token_ids = np.empty((num_sentences, max_length))

    for idx, sentence in enumerate(reviews):
        # Tokenize the sentence and map tokens to their corresponding IDs in the review_dict
        word_ids = [review_dict.get(token, 0) for token in nltk.word_tokenize(sentence)]
        if len(word_ids) > max_length:
            # If the sequence is longer than max_length, truncate it to max_length
            token_ids[idx] = word_ids[:max_length]
        else:
            # If the sequence is shorter than max_length, pad it with zeros to match max_length
            token_ids[idx] = word_ids + [0] * (max_length - len(word_ids))

    return token_ids


def get_embedding_map(review_dict, word_dict):
    """
    Creates an embedding map using the vocabulary dictionary and GloVe word embeddings.

    Args:
        review_dict (dict): The vocabulary dictionary containing word-to-ID mapping.
        word_dict (dict): The GloVe word embeddings dictionary containing word-to-embedding mapping.

    Returns:
        torch.Tensor: An embedding lookup table as a PyTorch tensor.
    """
    num_tokens = len(review_dict) + 2
    lookup_table = torch.empty((num_tokens, embedding_dim))

    for token_id in sorted(review_dict.values()):
        # Check if the token ID exists in the word embeddings dictionary
        if token_id in word_dict:
            lookup_table[token_id] = word_dict[token_id]
        else:
            # Use a default embedding of ones for unknown tokens
            lookup_table[token_id] = torch.ones((1, embedding_dim))

    # Use a zero embedding for the padding token
    lookup_table[0] = torch.zeros((1, embedding_dim))

    return lookup_table


def process():
    print('loading data')
    train_data, val_data, test_data = data_load.load_data()

    x_train_orig, y_train = train_data["text"].values, torch.LongTensor(train_data["label"].values).to(device)
    x_val_orig, y_val = val_data["text"].values, torch.LongTensor(val_data["label"].values).to(device)
    x_test_orig, y_test = test_data["text"].values, torch.LongTensor(test_data["label"].values).to(device)

    token_vocab, sequence_length = tokenize_reviews(x_train_orig, x_test_orig, x_val_orig)
    print(f"vocabulary tokens - {len(token_vocab)}, max seq length - {sequence_length}")
    print('loading word embeddings')
    word_embeddings = load_word_embeddings(token_vocab)
    print('completed loading word embeddings')

    x_train = torch.LongTensor(reviews_to_tokenid_map(x_train_orig, token_vocab, sequence_length)).to(device)
    x_test = torch.LongTensor(reviews_to_tokenid_map(x_test_orig, token_vocab, sequence_length)).to(device)
    x_val = torch.LongTensor(reviews_to_tokenid_map(x_val_orig, token_vocab, sequence_length)).to(device)

    # Training Preparation
    model = LSTMModule(len(token_vocab), sequence_length).to(device)
    lookup_table = get_embedding_map(token_vocab, word_embeddings)
    model.embedding.weight.data.copy_(lookup_table)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_accuracy = 0
    print("starting training")
    for epoch in range(n_epochs):
        train_loss = train_model(model, optimizer, criterion, x_train, y_train)
        train_accuracy = calculate_accuracy(x_train, y_train, model)
        validation_accuracy, _ = evaluate_model(model, x_val, y_val)
        print(f"\nepoch # {epoch}, train loss {train_loss}, "
              f"train accuracy {train_accuracy}, val accuracy {validation_accuracy}")

        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            torch.save(model.state_dict(), model_save_file_name)
            print(f"The model {model_save_file_name} has been saved!")
    print("training completed")

    print(model)

    # model Test
    test_model(model, x_test, y_test)


if __name__ == '__main__':
    process()
