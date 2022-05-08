import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import itertools

# My modules
from image_dataset import ImageDataset
import models
import augment
from plots import plot_roc_auc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Hyper Parameters
do_augment = True # Use augmented training data instead of original training data
num_epochs = 20
cnn_model_params = [{'in_channels': 3,   # must match 'out channels' of previous layer
                     'out_channels': 16, # can be anything
                     'kernel_size': 5,   # best to use an odd number (see padding)
                     'padding': 2,       # For padding = (kernel_size-1)/2 the dimensions will not change
                     'maxpool_size': 2,  # Pooling will reduce image dimensions by a factor of 'maxpool_size'
                     'dropout_rate': 1}, # 1 means no dropout
                    {'in_channels': 16,  # must match 'out channels' of previous layer
                     'out_channels': 32,
                     'kernel_size': 5,
                     'padding': 2,
                     'maxpool_size': 2,
                     'dropout_rate': 0.6}
                    ]
# Tuning hyper-parameters
# To run multiple parameter sets define each parameter vector values
# The program will loop over all combinations of these parameters and chose the
# One with the highest score.
# To run a fixed configuration set each vector to have one value only.
batch_size_list = [100]
learning_rate_list = [0.01]
optimizer_lambda_list = [0.6]
image_dim_list = [48]
kernel_size_list = [5]
dropout_rate_list = [0.6]

# Generate a list of all combinations of the tuning parameters
iteration_params = itertools.product(batch_size_list, learning_rate_list, optimizer_lambda_list, image_dim_list, kernel_size_list, dropout_rate_list)
iteration_params = [x for x in iteration_params]

result_list = []
best_f1 = 0
# The program expects the data path as an argument, the folders test/ and train/
# Should be located under this path.
parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('data_folder',type=str, help='Input folder path, containing images')

if __name__ == '__main__':
    args = parser.parse_args()
    train_data_dir = os.path.join(args.data_folder, 'train')
    test_data_dir = os.path.join(args.data_folder, 'test')
    # If augmentation is used it will be done into this directory
    augmented_data_dir = './augmented'
    # main loop- run over all parameter combinations
    for p in iteration_params:
        # decode the iterated parameters for the current iteration
        batch_size = p[0]
        learning_rate = p[1]
        optimizer_lambda = p[2]
        image_dim = p[3]
        cnn_model_params[0]['kernel_size'] = p[4]
        cnn_model_params[1]['kernel_size'] = p[4]
        cnn_model_params[0]['padding'] = (p[4]-1)//2
        cnn_model_params[1]['padding'] = (p[4] - 1) // 2
        cnn_model_params[1]['dropout_rate'] = p[5]

        if do_augment:
            data_dir = augmented_data_dir
            # Augment data ans copy all augmented data into a temporary directory for training
            augment.augment_files(train_data_dir, augmented_data_dir)
        else:
            data_dir = train_data_dir
        #
        # Initialize the image dataset
        train_dataset = ImageDataset(data_dir, 1, image_dim)

        # Initialize the image dataset
        test_dataset = ImageDataset(test_data_dir, 1, image_dim)

        # Data Loader (Input Pipeline)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False)
        # Instantiate the model
        model = models.CNN_params(cnn_model_params, image_dim).cuda(device)
        # Instantiate loss function
        criterion = nn.NLLLoss()
        # Instantiate optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # Instantiate scheduler
        lmbda = lambda epoch: optimizer_lambda ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lmbda)

        print('number of parameters: ', sum(param.numel() for param in model.parameters()))

        # Train the model
        do_train = 1
        res = models.train_test(model, num_epochs, {'train':train_loader, 'test': test_loader}, criterion, optimizer, scheduler, device, do_train)
        # Get all labels and predictions (softmax values) obtained during training
        train_labels = torch.concat(res[0]['train'])
        train_predictions = torch.concat(res[1]['train'])
        # Loss per epoch vector
        losses = res[2]
        # F1 per epoch vector
        f1s = res[3]
        # Plot Loss as a function of epoch number for test and train
        plt.title("Loss")
        plt.plot(np.array(range(len(losses['train']))) + 1, losses['train'], color="red")
        plt.plot(np.array(range(len(losses['test']))) + 1, losses['test'], color="blue")
        plt.legend(['Train loss', 'Test loss'])
        plt.xlabel('Ephoc #')
        plt.ylabel('Loss')
        plt.show()
        # Plot F1 as a function of epoch number for test and train
        plt.title("F1 Metric")
        plt.plot(np.array(range(len(f1s['train']))) + 1, f1s['train'], color="red")
        plt.plot(np.array(range(len(f1s['test']))) + 1, f1s['test'], color="blue")
        plt.legend(['Train F1', 'Test F1'])
        plt.xlabel('Ephoc #')
        plt.ylabel('F1')
        plt.show()
        # Plot ROC and calculate AUC for training data
        plot_roc_auc(train_labels.numpy(), train_predictions.numpy(), 'Train ROC')

        # Test the model
        print("\nTesting the model")
        # Instantiate the model
        model = models.CNN_params(cnn_model_params, image_dim)
        # Load the best state weights saved during training
        model.load_state_dict(torch.load('cnn.pkl'))
        model.cuda(device)
        do_train = 0 # Run with testing only
        res = models.train_test(model, num_epochs, {'test':test_loader}, criterion, optimizer, scheduler, device,
                                do_train)
        # Get all labels and predictions (softmax values) obtained during testing
        test_labels = torch.concat(res[0]['test'])
        test_predictions = torch.concat(res[1]['test'])
        # Plot ROC and calculate AUC for test data
        plot_roc_auc(test_labels.numpy(), test_predictions.numpy(), 'Test ROC')
        # Calculate F1 on test data, compare to the best achieved score
        # Make sure the parameters of the best model are saved
        f1s = res[3]['test'][0]
        if best_f1 < f1s:
            best_f1 = f1s
            best_p = p
            # Save the latest best model
            # After running all combinations of parameters the best will be saved
            torch.save(model.state_dict(), 'cnn_best.pkl')

    print("Run complete, result list:")
    print(f'Best iteration: batch_size = {best_p[0]}, learning_rate = {best_p[1]}, optimizer_lambda = {best_p[2]}, image_dim = {best_p[3]}, kernel_size = {best_p[4]},dropout_rate = {best_p[5]}')
    print(f'Score: %.2f %%' % (100.0*best_f1))
    # If augmentation was used- clean up the temporary directory, delete all files and remove folder
    if do_augment:
        print(f'Deleting augmenting files')
        if os.path.isdir(augmented_data_dir):
            files = os.scandir(path=augmented_data_dir)
            for f in files:
                os.remove(f)
            os.rmdir(augmented_data_dir)
            print(f'augmenting directory removed')
