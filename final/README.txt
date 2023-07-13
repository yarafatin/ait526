Problem Statement and solution Outline:
This project focuses exclusively on text-based reviews extracted from Yelp.com
The dataset used for our project is the Yelp reviews polarity dataset sourced from Kaggle.
The solution comprises models developed using LSTM, DistilBert, ELECTRA, and ensemble.
Data is cleaned by tokenization, stopwords removed and lemmatized using NLTK package.
For LSTM, GloVe embeddings is used, and passed through multiple layers.
The transformer models uses similar approach for data cleanup and trained.

System Minimum Requirements:
Windows 11 WSL2 Ubuntu 20.04
16-Core Processor  3.40 GHz
64GB Memory
Nvidia GTX 3090 GPU
The system was developed in AMD Ryzen 9 5950X machine. Intel architecture should also work fine
Latest GPU like GTX 3090 is needed for training
CPU based training is very slow and may take many days. It may not work

Installation:
1. Install Anaconda and create a new environment
2. Installing Cuda 12.1 may not be necessary because Pytorch includes it. If there are errors, install using below link
https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html
3. Install cuDNN 12.1 only if there are issue. Use below instructions but use version 12.1 though
https://medium.com/analytics-vidhya/installing-cuda-and-cudnn-on-windows-d44b8e9876b5
4. Install following packages
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch-nightly -c nvidia
conda install nltk scikit-learn pandas
conda install -c conda-forge transformers
5. Unzip the project source file to any location

Run Instructions:
All four models can be trained and tested by calling main.py
python main.py
You can also individually run LSTM model by executing python lstm_model.py
and transformer as python transformers_model.py
The attached log file - run.log shows the full code run on the full dataset



