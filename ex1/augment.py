import os
from torchvision.io import read_image
from torchvision.io import write_jpeg
from torchvision.transforms.functional import hflip, gaussian_blur
from torchvision import transforms

# Read image files from train_data_dir, generate 4 images from each image
# by augmentation (3 augmented images + the original imag) and save all 4 augmented
# images as jpeg files into augmented_data_dir
def augment_files(train_data_dir, augmented_data_dir):
    # Check if directory exist and if it does- remove all files
    print(f'Transferring and augmenting files into {augmented_data_dir}')
    if os.path.isdir(augmented_data_dir):
        files = os.scandir(path=augmented_data_dir)
        for f in files:
            os.remove(f)
    else:
        os.makedirs(augmented_data_dir)
    files = list(os.scandir(path=train_data_dir))
    t = transforms.Grayscale(3)
    num_files = len(files)
    i = 1
    prev_percentage = 0
    for f in files:
        if f.is_file():
            # Track progress
            percentage = i * 100 // num_files
            if percentage > prev_percentage:
                prev_percentage = percentage
                print(f'\r{percentage}%', end='')
            # Read original image from train_data_dir
            image = read_image(os.path.join(train_data_dir, f.name))
            new_filename = f.name
            new_filename1 = '1' + new_filename
            new_filename2 = '2' + new_filename
            new_filename3 = '3' + new_filename
            write_jpeg(image, os.path.join(augmented_data_dir, f.name))
            # Perform augmentations
            # Flip (left-right)
            image1 = hflip(image)
            # Add noise
            image2 = gaussian_blur(image, 5)
            # Remove colors
            image3 = t(image)
            write_jpeg(image1, os.path.join(augmented_data_dir, new_filename1))
            write_jpeg(image2, os.path.join(augmented_data_dir, new_filename2))
            write_jpeg(image3, os.path.join(augmented_data_dir, new_filename3))
            i = i + 1
    print(f'\nTransferred and augmented {i - 1} files')

if __name__ == '__main__':
    augment_files('./data/train', './data/augmented')