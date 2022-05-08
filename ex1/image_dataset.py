import os
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, img_dir, is_labeled, image_dim, transform=None, target_transform=None):
        files = os.scandir(path=img_dir)
        files_list = []
        self.labels = []
        self.image_dim = image_dim
        # Only store file names as a list because images are too big to store
        # The image will be loaded in __getitem__ using an index according to the order of this list
        for f in files:
            if f.is_file():
                files_list.append(f.name)
                # For labeled data the label is in the file name
                if is_labeled:
                    no_extention = f.name.split('.')[0]
                    label = 1 if (no_extention.split('_')[1] == '1') else 0
                self.labels.append(label)
        self.images = files_list
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = read_image(os.path.join(self.img_dir, self.images[idx]))
        image = transforms.Resize([self.image_dim, self.image_dim])(image).float()

        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label