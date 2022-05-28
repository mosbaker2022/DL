import torch
import torchtext
import argparse
import os
import models
from text_dataset import TextDataset
from torch.optim import Adam
import train_test

# Read data from data file into sentences.


# Hyper parameters
lstm_num_layers = 1
lstm_hidden_size = 300
lstm_bidirectional = True
lstm_drop_rate = 0.5
lstm_sequence_len = 40
loss_criterion = torch.nn.CrossEntropyLoss()
optimizer_lr = 0.001
num_epochs = 5
batch_size = 20
max_allowed_seq = 100 # maximum allowed tokens in one line

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# The program expects the data path as an argument, the folders test/ and train/
# Should be located under this path.

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_location', type = str, default='C:/Einan/Study/Courses/097200/Homework/HW2/data/', help='Input file location')
    args = parser.parse_args()
    train_data_dir = os.path.join(args.data_location, 'trainEmotions.csv')
    test_data_dir = os.path.join(args.data_location, 'testEmotions.csv')
    vec = torchtext.vocab.GloVe()
    # Sentences contains the data, tags contain the labels as string, label map contains a dictionary that maps label
    # strings to class number
    train_dataset = TextDataset(train_data_dir, vec, max_allowed_seq)
    test_dataset = TextDataset(test_data_dir, vec, max_allowed_seq)

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
                        seq_len = lstm_sequence_len, bidir = lstm_bidirectional, drop_rate = lstm_drop_rate)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=optimizer_lr)
    train_test.train(model, optimizer, device, loss_criterion, train_loader, test_loader, num_epochs)