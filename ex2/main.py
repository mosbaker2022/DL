import torch
import gensim
import gensim.downloader as gensim_api
from gensim.models import KeyedVectors
import argparse
import os
import models
from text_dataset import TextDataset
from torch.optim import Adam
import train_test
import numpy as np
import matplotlib.pyplot as plt

# Read data from data file into sentences.


# Hyper parameters
lstm_num_layers = 2
lstm_hidden_size = 100
lstm_bidirectional = True
lstm_drop_rate = 0.2
loss_criterion = torch.nn.CrossEntropyLoss()
optimizer_lr = 0.001
num_epochs = 1
batch_size = 64
max_allowed_seq = 64 # maximum allowed tokens in one line
tokenizer_use_lowercase = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# The program expects the data path as an argument, the folders test/ and train/
# Should be located under this path.

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_location', type = str, default='C:/Einan/Study/Courses/097200/Homework/HW2/data/', help='Input file location')
    args = parser.parse_args()
    train_data_dir = os.path.join(args.data_location, 'trainEmotions.csv')
    test_data_dir = os.path.join(args.data_location, 'testEmotions.csv')

    # Check if the vocabulary embedding model was saved before, if it was just
    # load it. If it wasn't, generate it from gensim and save it for the next time
    if os.path.isfile('word_vectors.kv'):
        print('Found local word_vectors file- Loading ...')
        word_vectors = KeyedVectors.load('word_vectors.kv')
    else:
        print('Local word2vec_vectors file not found- Downloading from gensim...')
        print('This may take a minute....')
        #word_vectors = gensim_api.load("glove-twitter-25")
        word_vectors = gensim_api.load("glove-twitter-100")
        #word_vectors = gensim_api.load('word2vec-google-news-300')
        #word_vectors.save('word2vec_vectors.kv')
        word_vectors.save('word_vectors.kv')
    # Designate a special vector (0) for any word out of the vocabulary

    v0 = np.zeros(word_vectors.vector_size)

    # use torchtext if available
    # vec = torchtext.vocab.GloVe()

    # Sentences contains the data, tags contain the labels as string, label map contains a dictionary that maps label
    # strings to class number
    train_dataset = TextDataset(train_data_dir, word_vectors,  {}, max_allowed_seq,
                                tokenizer_use_lowercase)
    test_dataset = TextDataset(test_data_dir, word_vectors, train_dataset.label_map, max_allowed_seq,
                               tokenizer_use_lowercase)

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    input_dim = test_dataset.sentences[0].shape[1]
    num_classes = len(train_dataset.label_map)
    # Generate the model object
    model = models.LSTM(input_size=input_dim, hidden_size=lstm_hidden_size, num_classes = num_classes, num_layers = lstm_num_layers,
                        bidir = lstm_bidirectional, drop_rate = lstm_drop_rate)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=optimizer_lr)
    results = train_test.train(model, optimizer, device, loss_criterion, train_loader, test_loader, num_epochs)
    train_loss_list = results[0]
    valid_loss_list = results[1]
    train_accuracy_list = results[2]
    test_accuracy_list = results[3]

    # Plot Loss as a function of epoch number for test and train
    plt.title("Loss")
    plt.plot(np.array(range(len(train_loss_list))) + 1, train_loss_list, color="red")
    plt.plot(np.array(range(len(valid_loss_list))) + 1, valid_loss_list, color="blue")
    plt.legend(['Train loss', 'Test loss'])
    plt.xlabel('Ephoc #')
    plt.ylabel('Loss')
    plt.show()
    # Plot accuracy as a function of epoch number for test and train
    plt.title("Accuracy Metric")
    plt.plot(np.array(range(len(train_accuracy_list))) + 1, train_accuracy_list, color="red")
    plt.plot(np.array(range(len(test_accuracy_list))) + 1, test_accuracy_list, color="blue")
    plt.legend(['Train Accuracy', 'Test Accuracy'])
    plt.xlabel('Ephoc #')
    plt.ylabel('Accuracy')
    plt.show()