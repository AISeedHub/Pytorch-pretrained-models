from torch.utils.data import Dataset
from torchvision import transforms
import torch
from torchvision.transforms._presets import ObjectDetection
from torchvision.transforms._presets import ImageClassification
from functools import partial
import numpy as np
import os
import json
import PIL.Image as Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


def normalize_image(image):
    xmin = np.min(image)
    xmax = np.max(image)
    return (image - xmin)/(xmax - xmin + 10e-6)

def collate_fn(batch):
    return tuple(zip(*batch))

class Standardize(object):
    """ Standardizes a 'PIL Image' such that each channel
        gets zero mean and unit variance. """
    def __call__(self, img):
        return (img - img.mean(dim=(1,2), keepdim=True)) \
            / torch.clamp(img.std(dim=(1,2), keepdim=True), min=1e-8)

    def __repr__(self):
        return self.__class__.__name__ + '()'



class ClassificationDataset(Dataset):
    '''
    mode: 'train', 'val', 'test'
    data_path: path to data folder
    imgsize: size of image
    transform: transform function
    '''
    portion = {
            "train": [0, 0.8],
            "val": [0.8, 0.9],
            "test": [0.9, 1]
        }
    def __init__(self, mode, data_path, imgsize=224, transform=None):
        super().__init__()
        self.transform = transform
        self.imgsize = imgsize
        self.data_path = data_path
        self.mode = mode
        self.transform = transform
        self.img_list = []
        self.label_list = []

        self.class_list = os.listdir(data_path)
        self.class_list.sort()
        self.num_class = len(self.class_list)


        for i in range(self.num_class):
            img_list = os.listdir(os.path.join(data_path, self.class_list[i]))
            data_range_below = int(ClassificationDataset.portion[mode][0] * len(img_list))
            data_range_above = int(ClassificationDataset.portion[mode][1] * len(img_list))
            self.img_list += [os.path.join(data_path, self.class_list[i], img) for img in img_list[data_range_below:data_range_above]]
            self.label_list += [i] * (data_range_above - data_range_below)

        if(self.transform is None):
            self.transform = ImageClassification(crop_size=imgsize)
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        '''
        return train_image, target, original image
        '''
        img_path = self.img_list[index]
        # check image_path exists
        if(not os.path.exists(img_path)):
            raise Exception("Image path does not exist")
        image = Image.open(img_path).convert("RGB")
        label = self.label_list[index]
        trans_img = self.transform(image)

        return trans_img, label, transforms.ToTensor()(image)

    
class DataModule(LightningDataModule):
    '''
    Data Module for Train/Val/Test data loadding
    Args: 
        data_settings, training_settings: hyperparameter settings
        transform: data augmentation
    Returns:
        Train/Test/Val data loader
    '''
    DATALOADER = {
        "ImageClassification": ClassificationDataset,
    }
    def __init__(self, data_settings, training_settings, transform=[None, None], collate_fn=None):
        super().__init__()

        assert data_settings['name'] in DataModule.DATALOADER.keys(), \
            "Data name should be one of {}".format(DataModule.DATALOADER.keys())

        self.dataset = DataModule.DATALOADER[data_settings['name']]
        self.root_dir = data_settings['path']
        self.img_size = data_settings['img_size']
        self.batch_size = training_settings['n_batch']
        self.num_workers = training_settings['num_workers']

        self.class_list = None
        self.collate_fn = collate_fn

        self.train_transform, self.val_transform = transform
        
    def setup(self, stage: str):

        if stage == "fit":
            self.Train_dataset = self.dataset(mode="train", data_path=self.root_dir,
                                                imgsize=self.img_size, transform=self.train_transform)
            self.Val_dataset = self.dataset(mode="val", data_path=self.root_dir,
                                                imgsize=self.img_size, transform=self.val_transform)
            
            self.class_list = self.Train_dataset.class_list
                
        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.Test_dataset = self.dataset(mode="test", data_path=self.root_dir,
                                                imgsize=self.img_size, transform=self.val_transform)
            
            self.class_list = self.Test_dataset.class_list
           
    def train_dataloader(self):
        return DataLoader(self.Train_dataset, batch_size=self.batch_size, shuffle=True, 
                          num_workers=self.num_workers, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.Val_dataset, batch_size=self.batch_size, shuffle=True, 
                          num_workers=self.num_workers, collate_fn=self.collate_fn)
    
    def test_dataloader(self):
        return DataLoader(self.Test_dataset, batch_size=self.batch_size, shuffle=False, 
                          num_workers=self.num_workers, collate_fn=self.collate_fn)

if __name__  == "__main__":

    bottle_dataset = DataModule(data_settings={"name": "ImageClassification", "path": "E:\code\AISeed\Bottle-Pi\dataset", "img_size": 224},
                                training_settings={"n_batch": 4, "num_workers": 4},
                                transform=[None, None])
    
    bottle_dataset.setup("fit")
    
    for bottle in bottle_dataset.train_dataloader():
        print(bottle)
        break
    print(bottle_dataset.class_list)
    