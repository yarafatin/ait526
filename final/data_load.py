"""
# AIT 526 - Natural Language Processing
# Final Project - Predicting Yelp Rating Polarity
# Group 9:
# Yasser Parambathkandy
# Indranil Pal
# 7/12/2023
"""
import pandas as pd
import nltk
import torch
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import os

from sklearn.model_selection import train_test_split

# Download and initialize required NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data():
    train = load_data_from_file('train.csv')
    test = load_data_from_file('test.csv')
    train, val = train_test_split(train, train_size=0.9, random_state=42)
    print(f"data length - train: {len(train)}, test:{len(test)}, validation:{len(val)}")
    return train, val, test


def load_data_from_file(file_name):
    """
    Loads the data from a CSV file. If the cleaned file already exists, it reads the dataframe from the file.
    Otherwise, it performs the cleaning steps, changes the label column, saves the cleaned dataframe to the output file,
    and returns the dataframe.

    Args:
        file_name (str): Name of the CSV file.
        label_as_tensor: update label column in dataframe as long tensor

    Returns:
        pandas.DataFrame: Cleaned dataframe.
    """
    # Check if the output file already exists
    output_file = 'data/cleaned_' + file_name
    if os.path.isfile(output_file):
        # If the file exists, read the dataframe from it
        # df = pd.read_csv(output_file, nrows=100)
        df = pd.read_csv(output_file)
    else:
        df = clean_and_save_file('data/' + file_name, output_file)
    return df


def clean_and_save_file(file_name, output_file):
    """
    Cleans the data in the CSV file, changes the label column from 1 to 0 and 2 to 1,
    saves the cleaned dataframe to the output file, and returns the dataframe.

    Args:
        file_name (str): Name of the CSV file.
        output_file (str): Name of the output CSV file.

    Returns:
        pandas.DataFrame: Cleaned dataframe.
    """
    # Load the CSV file
    df = pd.read_csv(file_name, names=['label', 'text'])
    print(f"rows in {file_name} original file is {len(df)}")
    # Clean the "text" column
    df['text'] = df['text'].apply(clean_text)
    # Remove rows with empty strings in the "text" column
    df = df[df['text'] != '']
    print(f"rows in {file_name} after cleanup is {len(df)}")
    # Change label column from 1 to 0 and 2 to 1
    df['label'] = df['label'].map({1: 0, 2: 1})
    # Save the cleaned dataframe to the output file
    df.to_csv(output_file, index=False)
    print(f"saving cleaned data to {output_file}")
    return df


def clean_text(text):
    """
    Cleans the text by removing stopwords, punctuation, and lemmatizing the words.

    Args:
        text (str): Input text.

    Returns:
        str: Cleaned text.
    """
    # Tokenize the text into words
    tokens = word_tokenize(text.lower())
    # Remove stopwords and punctuation
    tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]
    # Lemmatize the tokens
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    # Join the tokens back into a single string
    cleaned_text = ' '.join(tokens)
    return cleaned_text


if __name__ == '__main__':
    load_data()