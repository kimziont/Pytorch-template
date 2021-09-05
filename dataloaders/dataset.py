import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from skimage import io


class MaskDataset(Dataset):
    def __init__(self, kfold_annotations=False, kfold_label=False):
        transform = transforms.Compose([
                                            transforms.ToPILImage(),
                                            transforms.Resize((256, 256)),
                                            transforms.ToTensor(),
                                            transforms.ColorJitter(brightness=0.5),
                                            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
                                        ])
        self.kfold_annotations = kfold_annotations
        self.kfold_label = kfold_label
        if self.kfold_annotations is not False:
            self.annotations = kfold_annotations
            self.label = kfold_label
        else:
            self.annotations = pd.read_csv("/opt/ml/code/custom/data/shuffle_augmented_train_data.csv")
        self.root_dir = "/opt/ml/code/custom/data/shuff_aug_mask_train_data"
        self.transform = transform
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        
        if self.kfold_annotations is not False:
            img_path = os.path.join(self.root_dir, self.annotations.iloc[idx])
            label = torch.tensor(int(self.kfold_label.iloc[idx]))
        else:
            img_path = os.path.join(self.root_dir, self.annotations.loc[idx, 'file'])
            label = torch.tensor(int(self.annotations.loc[idx, 'label']))
        image = io.imread(img_path)
        if self.transform:
            image = self.transform(image)
        return (image, label)
    
    def get_dataframe(self):
        return self.annotations



class BasicDataset(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.image_df = X
        self.label_df = y
    
    def __len__(self):
        return len(self.image_df)
    
    def __getitem__(self, idx):
        return self.image_df.iloc[idx], self.label_df.iloc[idx]



