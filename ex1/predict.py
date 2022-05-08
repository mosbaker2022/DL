import os
import argparse
from torchvision.io import read_image
import torchvision.transforms as transforms
import pandas as pd
import models
import torch

image_dim = 48
cnn_model_params = [{'in_channels': 3,
                     'out_channels': 16,
                     'kernel_size': 5,
                     'padding': 2,
                     'maxpool_size': 2,
                     'dropout_rate': 1},
                    {'in_channels': 16,
                     'out_channels': 32,
                     'kernel_size': 5,
                     'padding': 2,
                     'maxpool_size': 2,
                     'dropout_rate': 0.6}
                    ]

# Parsing script arguments
# Run with one argument which is the path under which all test files are located
parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('input_folder',type=str, help='Input folder path, containing images')

if __name__ == '__main__':
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.CNN_params(cnn_model_params, image_dim)
    model.load_state_dict(torch.load('cnn_best.pkl'))
    model.cuda(device)
    model.eval()
# Reading input folder
    files = os.listdir(args.input_folder)
    predictions_list = []
    label_list = []
    prev_percentage = 0
    i=0
    for filename in files:
        percentage = i * 100 // len(files)
        if percentage > prev_percentage:
            prev_percentage = percentage
            print(f'\r{percentage}%', end='')
        i=i+1
        image = read_image(os.path.join(args.input_folder, filename))
        image = transforms.Resize([image_dim, image_dim])(image).float()
        image = image.to(device)
        image = torch.unsqueeze(image,0)
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        predictions_list.append(predicted.data.to('cpu').numpy()[0])
        no_extention = filename.split('.')[0]
        label = 1 if (no_extention.split('_')[1] == '1') else 0
        label_list.append(label)
    # Convert data to dataframe and save to a text file as CSV
    prediction_df = pd.DataFrame({'filename': files,
                           'prediction': predictions_list})
    prediction_df.to_csv("prediction.csv", index=False, header=False)
