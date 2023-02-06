from genericpath import isdir
import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset
import torchvision.models as models

from imageio import imsave

import random
import numpy as np
from PIL import Image

# utility functions
def clip_epsilon(x_adv, x, epsilon, min=0, max=1):
    x_adv = torch.where(x_adv > x + epsilon, x + epsilon, x_adv)
    x_adv = torch.where(x_adv < x - epsilon, x - epsilon, x_adv)
    x_adv = torch.clamp(x_adv, min=min, max=max)
    return x_adv

def set_seed(seed=0):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def save_img(x_adv, save_dir, filenames):
    os.makedirs(save_dir, exist_ok=True)
    for i, x_adv_single in enumerate(x_adv):
        im = T.ToPILImage()(x_adv_single)
        final_path = os.path.join(save_dir, filenames[i])
        # im.save(final_path, format='png')
        imsave(final_path, im, format='png')
    print('{} adv. examples saved'.format(i+1))  

class SelectedImagenet(Dataset):
    def __init__(self, imagenet_val_dir, selected_images_csv, transform=None):
        super(SelectedImagenet, self).__init__()
        self.imagenet_val_dir = imagenet_val_dir
        self.selected_images_csv = selected_images_csv
        self.transform = transform
        self.selected_list = []
        # labels
        folders = os.listdir(imagenet_val_dir)
        self.label_to_index = {name: i for i, name in enumerate(folders)}
        # load images from csv
        self._load_csv()

    def _load_csv(self):
        reader = csv.reader(open(self.selected_images_csv, 'r'))
        #next(reader)
        for row in reader:
            label = row[0]
            filepaths = row[1:]
            for item in filepaths:
                if item != '':
                    self.selected_list.append((item, label))

    def __getitem__(self, item):
        filepath, label = self.selected_list[item]
        image = Image.open(os.path.join(self.imagenet_val_dir, label, filepath))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, self.label_to_index[label]

    def __len__(self):
        return len(self.selected_list)


# Similar to SelectedImagenet, but it will only read the first column and select all files in the folders
class SelectedImagenetFolder(Dataset):
    def __init__(self, imagenet_val_dir, selected_images_file, transform=None):
        super(SelectedImagenetFolder, self).__init__()
        self.imagenet_val_dir = imagenet_val_dir
        self.selected_images_file = selected_images_file
        self.transform = transform
        self.selected_list = []
        # labels
        folders = os.listdir(imagenet_val_dir)
        self.label_to_index = {name: i for i, name in enumerate(folders)}
        # load images from csv
        self._load_csv()

    def _load_csv(self):
        with open(self.selected_images_file, 'r') as f:
            folders = f.readlines()
        for row in folders:
            label = row.split(',')[0]
            for img in os.listdir(os.path.join(self.imagenet_val_dir, label)):
                self.selected_list.append((img, label))
        print("[SelectedImagenetFolder] Found {} images in {} folders".format(len(self.selected_list), len(folders)))

    def __getitem__(self, item):
        filepath, label = self.selected_list[item]
        image = Image.open(os.path.join(self.imagenet_val_dir, label, filepath))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, self.label_to_index[label]

    def __len__(self):
        return len(self.selected_list)

class SelectedCifar(Dataset):
    def __init__(self, path, labels, transform=None):
        super(SelectedCifar, self).__init__()
        self.path = path        
        # load data. Dir => np format. Pt => torch format.
        if os.path.isdir(path):
            files = os.listdir(path)
            x_list = []
            for f in range(len(files)-1):
                x_list.append(np.load(os.path.join(path, 'batch_{}.npy'.format(f))) / 255.0)
            self.x = np.concatenate(x_list, axis=0)
            self.labels = np.load(os.path.join(path, 'labels.npy'))
        else:
            self.x = torch.load(path).squeeze(1)
            self.labels = torch.load(labels)

    def __getitem__(self, idx):
        image = self.x[idx]
        label = self.labels[idx]
        #if self.transform is not None:
        #    image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.x)

class DuoImageFolders(Dataset):
    '''
    [ILA-DA] For validation on defended models. \\
    Load from two folders, one for clean images and one for reference attacks.
    '''
    def __init__(self, transform, clean_dir, atk_dir, label_dir='data/val_rs.csv'):
        super(DuoImageFolders, self).__init__()
        self.transform = transform
        self.clean_dir = clean_dir
        self.atk_dir = atk_dir
        self.clean_examples = self.__read_dir__(clean_dir)
        self.adv_examples = self.__read_dir__(atk_dir)        
        self.path_label_dict = self.__read_label__(label_dir)
        assert len(self.clean_examples) == len(self.adv_examples)        
        assert abs(len(self.clean_examples) - len(self.path_label_dict)) <= 1, "{} <-> {}".format(len(self.clean_examples), len(self.path_label_dict)) 
        print("[DuoImageFolders] Found {} pair of images (clean, adv, label)".format(len(self.clean_examples)))       

    def __read_dir__(self, dir):
        assert os.path.exists(dir)
        out_list = []
        for f in os.listdir(dir):
            if not os.path.isfile(os.path.join(dir, f)):
                continue
            out_list.append(f)
        return out_list
    
    def __read_label__(self, label_path):
        path_label_dict = {}
        with open(label_path, 'r') as f:
            path_label_pairs = f.readlines()
            for row in path_label_pairs:
                path, label = row.strip().split(',')
                if not label.isdigit():
                    continue
                # assume the label must be int
                path_label_dict[path] = int(label)
        return path_label_dict

    def __getitem__(self, item):
        filename = self.adv_examples[item]
        label = self.path_label_dict[filename]
        clean_example = Image.open(os.path.join(self.clean_dir, filename))
        adv_example = Image.open(os.path.join(self.atk_dir, filename))
        if clean_example.mode != 'RGB':
            clean_example = clean_example.convert('RGB')
        if adv_example.mode != 'RGB':
            adv_example = adv_example.convert('RGB')            
        if self.transform is not None:
            clean_example = self.transform(clean_example)
            adv_example = self.transform(adv_example)
        return clean_example, adv_example, filename, label

    def __len__(self):
        return len(self.clean_examples)

class Normalize(nn.Module):
    def __init__(self, ms=None):
        super(Normalize, self).__init__()
        if ms is None:
            self.ms = [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]
        else:
            self.ms = ms

    def forward(self, input):
        x = input.clone()
        for i in range(x.shape[1]):
            x[:,i] = (x[:,i] - self.ms[0][i]) / self.ms[1][i]
        return x

def unnormalize(image):
    mean = torch.tensor((0.485, 0.456, 0.406))
    std = torch.tensor((0.229, 0.224, 0.225))
    return T.Normalize((-mean / std), (1.0 / std))(image)


# ILA
def ila_forw_resnet50(model, x, ila_layer):
    jj = int(ila_layer.split('_')[0])
    kk = int(ila_layer.split('_')[1]) - 1
    x = model[0](x)
    x = model[1].conv1(x)
    x = model[1].bn1(x)
    x = model[1].relu(x)
    if jj == 0 and kk == 0:
        return x
    x = model[1].maxpool(x)

    for ind, mm in enumerate(model[1].layer1):
        x = mm(x)
        if jj == 1 and ind == kk:
            return x
    for ind, mm in enumerate(model[1].layer2):
        x = mm(x)
        if jj == 2 and ind == kk:
            return x
    for ind, mm in enumerate(model[1].layer3):
        x = mm(x)
        if jj == 3 and ind == kk:
            return x
    for ind, mm in enumerate(model[1].layer4):
        x = mm(x)
        if jj == 4 and ind == kk:
            return x
    raise Exception('ResNet reaches its end. Cannot find layer {}'.format(ila_layer))

def ila_forw_inception_v3(model, x, ila_layer):    
    norm, model = model[0], model[1]
    x = norm(x)
    x = model.Conv2d_1a_3x3(x)
    if ila_layer == '1a': return x
    x = model.Conv2d_2a_3x3(x)
    if ila_layer == '2a': return x
    x = model.Conv2d_2b_3x3(x)
    if ila_layer == '2b': return x
    x = model.maxpool1(x)
    x = model.Conv2d_3b_1x1(x)
    if ila_layer == '3b': return x
    x = model.Conv2d_4a_3x3(x)
    if ila_layer == '4b': return x
    x = model.maxpool2(x)
    x = model.Mixed_5b(x)
    if ila_layer == '5b': return x
    x = model.Mixed_5c(x)
    if ila_layer == '5c': return x
    x = model.Mixed_5d(x)
    if ila_layer == '5d': return x
    x = model.Mixed_6a(x)
    if ila_layer == '6a': return x
    x = model.Mixed_6b(x)
    if ila_layer == '6b': return x
    x = model.Mixed_6c(x)
    if ila_layer == '6c': return x
    x = model.Mixed_6d(x)
    if ila_layer == '6d': return x
    x = model.Mixed_6e(x)
    if ila_layer == '6e': return x
    x = model.Mixed_7a(x)
    if ila_layer == '7a': return x
    x = model.Mixed_7b(x)
    if ila_layer == '7b': return x
    x = model.Mixed_7c(x)
    if ila_layer == '7c': return x
    raise Exception('Inception V3 reaches its end. Cannot find layer {}'.format(ila_layer))

def ila_forw_adv_inception_v3(model, x, ila_layer):
    norm, model = model[0], model[1]
    #x = norm(x)
    x = model.Conv2d_1a_3x3(x)
    if ila_layer == '1a': return x
    x = model.Conv2d_2a_3x3(x)
    if ila_layer == '2a': return x
    x = model.Conv2d_2b_3x3(x)
    if ila_layer == '2b': return x
    x = model.Pool1(x)
    x = model.Conv2d_3b_1x1(x)
    if ila_layer == '3b': return x
    x = model.Conv2d_4a_3x3(x)
    if ila_layer == '4b': return x
    x = model.Pool2(x)
    x = model.Mixed_5b(x)
    if ila_layer == '5b': return x
    x = model.Mixed_5c(x)
    if ila_layer == '5c': return x
    x = model.Mixed_5d(x)
    if ila_layer == '5d': return x
    x = model.Mixed_6a(x)
    if ila_layer == '6a': return x
    x = model.Mixed_6b(x)
    if ila_layer == '6b': return x
    x = model.Mixed_6c(x)
    if ila_layer == '6c': return x
    x = model.Mixed_6d(x)
    if ila_layer == '6d': return x
    x = model.Mixed_6e(x)
    if ila_layer == '6e': return x
    x = model.Mixed_7a(x)
    if ila_layer == '7a': return x
    x = model.Mixed_7b(x)
    if ila_layer == '7b': return x
    x = model.Mixed_7c(x)
    if ila_layer == '7c': return x
    raise Exception('Inception V3 reaches its end. Cannot find layer {}'.format(ila_layer))


def ila_forw_vgg(model, x, ila_layer):
    norm, model = model[0], model[1]
    x = norm(x)
    layer_cnt = 0
    for i, layer in enumerate(model.features):
        x = layer(x)
        if isinstance(layer, nn.Conv2d):
            layer_cnt += 1
            if layer_cnt == ila_layer:
                return x
    raise Exception('The VGG model reaches its end. Cannot find layer {}'.format(ila_layer))

class ILAProjLoss(torch.nn.Module):
    def __init__(self):
        super(ILAProjLoss, self).__init__()
    def forward(self, old_attack_mid, new_mid, original_mid, coeff):
        n = old_attack_mid.shape[0]
        x = (old_attack_mid - original_mid).reshape(n, -1)
        y = (new_mid - original_mid).reshape(n, -1)        
        # x_norm = x / torch.norm(x, dim = 1, keepdim = True)
        proj_loss = torch.sum(y * x) / n
        return proj_loss
 