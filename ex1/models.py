import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import  f1_score
# cnn model classifier, with variable number of cnn layers and changeable parameters
class CNN_params(nn.Module):
    def __init__(self, layer_params, image_dim):
        super(CNN_params, self).__init__()
        self.image_dim = image_dim
        self.num_layers = len(layer_params)
        self.layer = []
        # Generate multiple sequential cnn layers in a loop
        # Number of layers is determined by the number of elements in layer_params
        for k in range(self.num_layers):
            param = layer_params[k]
            self.layer.append(nn.Sequential(
                nn.Conv2d(param['in_channels'], param['out_channels'], kernel_size=param['kernel_size'], padding=param['padding']),
                nn.BatchNorm2d(param['out_channels']),
                nn.ReLU(),
                nn.MaxPool2d(param['maxpool_size'])))
            self.image_dim = self.image_dim/param['maxpool_size']

        # need to convert to ModuleList because a normal list will not perform all module
        # operations such as push to cuda etc.
        self.layer = nn.ModuleList(self.layer)

        # Add a fully connected layer for classification (2 classes))
        self.fc = nn.Linear(int((param['out_channels'] * self.image_dim * self.image_dim)), 2)
        # Dropout for regularization
        self.dropout = nn.Dropout(p=param['dropout_rate'])
        # Softmax to convert to probabilities
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = x
        for k in range(self.num_layers):
            out = self.layer[k](out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        return self.logsoftmax(out)

def train_test(model, num_epochs, loader, criterion, optimizer, scheduler, device, do_train):
    # Collect all labels in order for ROC calculation
    all_labels = {'train':[], 'test': []}
    # Collect all probabilities in order for ROC calculation
    all_softmax_data = {'train':[], 'test': []}
    # Collect loss and F1 results per epoch
    res_loss = {'train': [], 'test': []}
    res_f1 = {'train': [], 'test': []}
    # In training mode do train and test
    if do_train:
        epochs = num_epochs
        train_test_vect = ['train', 'test']
    else:
        # In test mode do test only, and one epoch only
        epochs = 1
        train_test_vect = ['test']
    best_res_f1 = 0
    for epoch in range(epochs):
        prev_percentage = 0
        for train_test in train_test_vect: # in test mode run only once, in train mode run twice (for training and testing)
            losses = []
            f1s = []
            # set model mode according to the proper operation
            if train_test == 'train':
                model.train()
            else:
                model.eval()
            for i, (images, labels) in enumerate(loader[train_test]):

                if epoch == (epochs-1): # only accumulate the results from the last epoch
                    all_labels[train_test].append(labels.to('cpu'))

                images = images.to(device)
                labels = labels.to(device)

                # Forward + Backward + Optimize
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                if epoch == (epochs - 1): # only accumulate the results from the last epoch
                    softmax_data = outputs.data[:,1].to('cpu')
                    all_softmax_data[train_test].append(softmax_data)

                if train_test == 'train': # in training also calculate loss and backprop
                    loss = criterion(outputs, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if (i + 1) % 10 == 0:
                        print('Epoch [%d/%d], Iter [%d/%d], Loss: %.4f'
                              % (epoch + 1, num_epochs, i + 1,
                                 len(loader[train_test]), loss.data))
                    # move results into CPU
                    losses.append(loss.data.to("cpu"))
                    f1s.append(f1_score(labels.to("cpu"), predicted.to("cpu")))

                else: # test only calculate metric and loss
                    loss = criterion(outputs, labels)
                    losses.append(loss.data.to("cpu"))
                    f1s.append(f1_score(labels.to("cpu"), predicted.to("cpu")))

                    # Print progress percentage
                    percentage = (i+1) * 100 // len(loader[train_test])
                    if (percentage > prev_percentage):
                        prev_percentage = percentage
                        print(f'\r{percentage}%', end='')
            if train_test == 'train':
                print('Train epoch average metrics loss: %.4f, f1: %.4f %%' % (np.average(np.array(losses)),
                  100.0*np.average(np.array(f1s))))
            else:
                print('\nTest average metrics loss: %.4f, f1: %.4f %%' % (np.average(np.array(losses)),
                      100.0*np.average(np.array(f1s))))
            # build the F1 and Loss result lists (one item per epoch)
            res_loss[train_test].append(np.average(np.array(losses)))
            res_f1[train_test].append(np.average(np.array(f1s)))
        if do_train:
            # Save the model state if it is better than best model found so far
            if res_f1['test'][-1] > best_res_f1:
                torch.save(model.state_dict(), 'cnn.pkl')
                best_res_f1 = res_f1['test'][-1]
            scheduler.step()
    # Return the final result: labels, softmax data, loss and f1 score lists
    return [all_labels, all_softmax_data, res_loss, res_f1]