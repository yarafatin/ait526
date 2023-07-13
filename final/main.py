"""
# AIT 526 - Natural Language Processing
# Final Project - Predicting Yelp Rating Polarity
# Group 9:
# Yasser Parambathkandy
# Indranil Pal
# 7/12/2023
"""

import lstm_model
import transformers_model

if __name__ == '__main__':
    """
    Train, validate, and test LSTM model
    """
    print('starting LSTM model')
    lstm_model.process()
    print('completed LSTM model')
    """
    Train, validate, and test DistillBert, ELECTRA, and ensemble models
    """
    print('starting Transformer model')
    transformers_model.process()
    print('completed Transformer model')
