import os
import random
import torch
import pickle
import numpy as np
from PIL import Image
import torch.utils.data as data

def unpickle(file):
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')


class CifarDataset(data.Dataset):
    def __init__(self, train=True, transform=None):
        self.images = []
        self.labels = []
        self.transform = transform
        self.noise = np.random.randint(-5, 5, size=(32, 32, 3))
        
        if train:
            for index in range(5):
                dataDict = unpickle(f'./cifar-10-batches-py/data_batch_{index + 1}')
                for index, lineData in enumerate(dataDict[b'data']):
                    img = np.reshape(lineData, (3, 32, 32))
                    img = np.transpose(img, (1, 2, 0))
                    self.images.append(img)
                    self.labels.append(dataDict[b'labels'][index])
        else:
            dataDict = unpickle(f'./cifar-10-batches-py/test_batch')
            for index, lineData in enumerate(dataDict[b'data']):
                img = np.reshape(lineData, (3, 32, 32))
                img = np.transpose(img, (1, 2, 0))
                self.images.append(img)
                self.labels.append(dataDict[b'labels'][index])

    def __getitem__(self, index):

        image = self.images[index]
        label = self.labels[index]
        
        if random.random() < 0.5:
            image = image + np.random.randint(-1, 2, size=1)[0] * self.noise
            image = np.uint8(image)

        image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.images)
        
