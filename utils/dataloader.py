import os
import cv2 
import torch
import numpy as np
from easydict import EasyDict as edict
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import yaml


def Decode_MPII(line):
    anno = edict()
    anno.face, anno.lefteye, anno.righteye = line[0], line[1], line[2]
    anno.name = line[3]

    anno.gaze3d, anno.head3d = line[5], line[6]
    anno.gaze2d, anno.head2d = line[7], line[8]
    return anno

def Decode_Diap(line):
    anno = edict()
    anno.face, anno.lefteye, anno.righteye = line[0], line[1], line[2]
    anno.name = line[3]

    anno.gaze3d, anno.head3d = line[4], line[5]
    anno.gaze2d, anno.head2d = line[6], line[7]
    return anno

def Decode_Gaze360(line):
    anno = edict()
    anno.face, anno.lefteye, anno.righteye = line[0], line[1], line[2]
    anno.name = line[3]

    anno.gaze3d = line[4]
    anno.gaze2d = line[5]
    return anno

def Decode_ETH(line):
    anno = edict()
    anno.face = line[0]
    anno.gaze3d = line[1]
    anno.gaze2d = line[2]
    anno.head3d = line[3] 
    anno.head2d = line[4]
    anno.name = line[5]
    return anno

def Decode_GazeCapture(line):
    anno = edict()
    anno.face = line[0]
    anno.gaze2d = line[1]
    anno.head2d = line[2]
    anno.name = line[3]
    return anno

def Decode_Dict() -> edict:
    mapping = edict()
    mapping.mpiigaze = Decode_MPII
    mapping.eyediap = Decode_Diap
    mapping.gaze360 = Decode_Gaze360
    mapping.eth = Decode_ETH
    mapping.gazecapture = Decode_GazeCapture
    return mapping

# For fuzzy search
def long_substr(str1, str2) -> int:
    substr = ''
    for i in range(len(str1)):
        for j in range(len(str1)-i+1):
            if j > len(substr) and (str1[i:i+j] in str2):
                substr = str1[i:i+j]
    return len(substr)


def Get_Decode(name):
    mapping = Decode_Dict()
    keys = list(mapping.keys())
    name = name.lower()
    score = [long_substr(name, i) for i in keys]
    key = keys[score.index(max(score))]
    return mapping[key]
    

class GazeDataset(Dataset): 
    def __init__(self, dataset:edict, augmentation:bool=False):

        self.augmentation = augmentation
        # Read source data
        self.source = edict() 
        self.source.line = []
        self.source.root = dataset.image
        self.source.decode = Get_Decode(dataset.name)

        if isinstance(dataset.label, list):
            for i in dataset.label:
                with open(i) as f: line = f.readlines()
                if dataset.header: line.pop(0)
                self.source.line.extend(line)
        else:
            with open(dataset.label) as f: self.source.line = f.readlines()
            if dataset.header: self.source.line.pop(0)

        # build transforms
        if augmentation:
            self.transforms = transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2), 
                transforms.ToTensor()
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor()
            ])

    def __len__(self): 
        return len(self.source.line)

    def __getitem__(self, idx):

        # Read souce information
        line = self.source.line[idx]
        line = line.strip().split(' ')
        anno = self.source.decode(line)

        img = cv2.imread(os.path.join(self.source.root, anno.face))
        img = self.transforms(img)

        label = np.array(anno.gaze3d.split(',')).astype('float')
        label = torch.from_numpy(label).type(torch.FloatTensor)

        data = edict()
        data.face = img
        data.name = anno.name

        return data, label

def loader(source, batch_size=1, shuffle=False, augmentation=False, num_workers=1, sampler=None):
    dataset = GazeDataset(source, augmentation)
    print(f'-- [Read Data]: Total num: {len(dataset)}')
    print(f'-- [Read Data]: Source: {source.label}')
    build_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, sampler=sampler)
    return build_loader


if __name__ == '__main__':
  
    # path = './p00.label'
    # d = loader(path)
    # print(len(d))
    # (data, label) = d.__getitem__(0)

    # print(Get_Decode('eth'))

    
    # (data, label) = dl[0]

    data = edict(yaml.load(open('/data/hanhc/GazeEstimation/GSAL_hhc/config/train/config_eth_train.yaml'), Loader=yaml.FullLoader)).data
    print(data)
    dl = loader(data, batch_size=64, shuffle=False, num_workers=16)
    print(len(dl))
    print(len(DataLoader(GazeDataset(data))))
    # for batch in dl:
    #     (data, label) = batch
    #     img = data.face[0, :, :, :]
    #     label = label[0, :]
    #     img = img.cuda()
    #     print(img)
    #     print('-', end='')
