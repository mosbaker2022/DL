import torch
from gensim.models import KeyedVectors
import argparse
import os
import models
from text_dataset import TextDataset
import numpy as np

# Hyper parameters
lstm_num_layers = 1
lstm_hidden_size = 64
lstm_bidirectional = True
lstm_drop_rate = 0.6
loss_criterion = torch.nn.CrossEntropyLoss()
optimizer_lr = 0.001
num_epochs = 10
batch_size = 64
max_allowed_seq = 64 # maximum allowed tokens in one line
use_embs = True
feeze_embs = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def predict(model, device, data_loader):
    model.eval()
    predictions = []
    for (data, labels, data_len) in data_loader:
        data = data.to(device)
        data_len = data_len.to('cpu')
        output = model(data, data_len)
        predicted_class = torch.argmax(output, 1)
        predictions += list(predicted_class.to('cpu'))
    return predictions

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_location', type = str, help='Input file location')
    args = parser.parse_args()
    test_data_dir = os.path.join(args.data_location)
    if os.path.isfile('word_vectors.kv'):
        print('Found local word_vectors file- Loading ...')
        word_vectors = KeyedVectors.load('word_vectors.kv')
    else:
        print("Missing local word_vectors file...")
        exit(-1)
    embs = word_vectors.vectors
    pad_vect = np.zeros((1, embs.shape[1]))  # pad '<pad>' is a zero vector
    unk_vect = np.mean(embs, axis=0, keepdims=True)  # unknown '<unk>' is the mean of all vectors
    embs = np.vstack((pad_vect, unk_vect, embs))  # insert pad and unk vectors at the beginning of the array

    label_map = {'happiness':0, 'sadness': 1, 'neutral': 2}

    test_dataset = TextDataset(test_data_dir, pad_vect, unk_vect, word_vectors, label_map,
                               max_allowed_seq,
                               use_embs)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    model = models.LSTM_emb(embs, feeze_embs=feeze_embs, hidden_size=lstm_hidden_size, num_classes=len(label_map),
                            num_layers=lstm_num_layers,
                            bidir=lstm_bidirectional, drop_rate=lstm_drop_rate)
    model.load_state_dict(torch.load('best_model.pkl'))
    model.to(device)
    predicted_labels = predict(model, device, test_loader)
    class_to_label = list(label_map.keys())
    with open(test_data_dir, 'r') as f:  # Read file into string
        lines_data = f.readlines()
        str_list = []
        n = 0
        str_list.append(lines_data[0])
        for line1 in lines_data[1:]:
            split_index = line1.find(',')
            lebel_str = class_to_label[predicted_labels[n]] # assign the label from predictions
            text_str = line1[split_index + 1:-1]
            str1 = ''.join([lebel_str+',', text_str+'\n'])
            str_list.append(str1)
            n += 1
    with open('prediction.csv', 'w') as f:  # write strings into file
        f.writelines(str_list)
    print('Done. Prediction written to file prediction.csv')
