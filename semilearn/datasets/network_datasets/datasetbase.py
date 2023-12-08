# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import copy
import numpy as np 
from PIL import Image
import torchvision
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from semilearn.datasets.utils import get_onehot, random_case_swap, random_mask

class NetworkDataAugmentation:
    def __init__(self, p=0.5):
        """Custom class to create data augmentation for network data.
        For now only consider alphabet character lower/upper swap and random masking.

        p(float): probability for random transformations. Defaults to 0.5
        """
        self.p = p
    
    def __call__(self, dat):
        aug_dat = random_case_swap(dat)
        #aug_dat = random_mask(dat)

        return aug_dat

class NetImgAugmentation:
    """
    NetImgAugmentation applies random_case_swap for upper/lower case alphabets and random mask, where
    p1 is the probability of applying random_case_swap and p2 is the probablity of applying random_mask.
    """
    def __init__(self, p1=0.3, p2=0.4):
        self.p1 = p1
        self.p2 = p2

    def __call__(self, dat: np.ndarray):
        if dat.ndim == 3:
            mask_shape = (dat.shape[0], dat.shape[1], 1)
        if dat.ndim == 4:
            mask_shape = (dat.shape[0], dat.shape[1], dat.shape[2], 1)
        # Random case swap
        if self.p1 is not None:

            lower_idx = np.where((97 < dat) & (dat <= 122))
            upper_idx = np.where((65 <= dat) & (dat <= 97))

            mask_lower = np.random.choice([True, False], size=len(lower_idx[0]), p=[self.p1, 1-self.p1])
            mask_upper = np.random.choice([True, False], size=len(upper_idx[0]), p=[self.p1, 1-self.p1])

            dat[lower_idx][mask_lower] += 32
            dat[upper_idx][mask_upper] -= 32

        # Random Mask
        mask = np.random.choice([0,1], size=mask_shape, p=[self.p2, 1 - self.p2])
        aug_dat = np.multiply(dat, mask)

        return aug_dat


class BasicDataset(Dataset):
    """
    BasicDataset returns a pair of payload_byte network data and labels (targets).
    If targets are not given, BasicDataset returns None as the label.
    This class supports strong augmentation for FixMatch, #TODO: AUGMENTATION FOR NETWORK TRAFFIC DATA?
    and return both weakly and strongly augmented images.
    """

    def __init__(self,
                 alg,
                 data,
                 targets=None,
                 num_classes=None,
                 is_ulb=False,
                 onehot=False,
                 *args, 
                 **kwargs):
        """
        Args:
            alg (_type_): _description_
            data (numpy.array): x_data
            targets (numpy.array, optional): y_data. If not exist, defaults to None.
            num_classes (int, optional): number of label classes. Defaults to None.
            is_ulb (bool, optional): Whether data is unlabeled. Defaults to False.
            onehot (bool, optional): If True, label is converted into onehot vector. Defaults to False.
        """        
        super(BasicDataset, self).__init__()
        self.alg = alg
        self.data = data
        self.targets = targets

        self.num_classes = num_classes
        self.is_ulb = is_ulb
        self.onehot = onehot
        self.transform = None #random_case_swap()
        self.strong_transform = NetworkDataAugmentation()
        if self.strong_transform is None:
            if self.is_ulb:
                assert self.alg not in ['fullysupervised', 'supervised', 'pseudolabel', 'vat', 'pimodel', 'meanteacher', 'mixmatch'], f"alg {self.alg} requires strong augmentation"
    
    def __sample__(self, idx):
        """ dataset specific sample function """
        # set idx-th target
        if self.targets is None:
            target = None
        else:
            target_ = self.targets[idx]
            target = target_ if not self.onehot else get_onehot(self.num_classes, target_)

        # set augmented images
        data = self.data[idx]
        return data, target

    def __getitem__(self, idx):
        """
        If strong augmentation is not used,
            return weak_augment_data, target
        else:
            return weak_augment_data, strong_augment_data, target
        """
        if self.targets is None:
            target = None
        else:
            target_ = self.targets[idx]
            target = target_ if not self.onehot else get_onehot(self.num_classes, target_)

        dat = self.data[idx]

        if self.is_ulb == False:
            return {'idx_lb':idx, 'x_lb':dat, 'y_lb':target} 
        else:
            if self.alg == 'fullysupervised' or self.alg == 'supervised':
                return {'idx_ulb':idx} 
            elif self.alg == 'pseudolabel' or self.alg == 'vat':
                return {'idx_ulb':idx, 'x_ulb_w':dat} 
            elif self.alg == 'pimodel' or self.alg == 'meanteacher' or self.alg == 'mixmatch':
                # Use vanilla x_data form weak augmentation
                dat_s = self.strong_transform(dat)
                return {'idx_ulb':idx, 'x_ulb_w':dat, 'x_ulb_s':dat_s}
            elif self.alg == 'comatch' or self.alg == 'remixmatch':
                dat_s = self.strong_transform(dat)
                dat_s_ = self.strong_transform(dat)
                return {'idx_ulb':idx, 'x_ulb_w': dat, 'x_ulb_s_0':dat_s, 'x_ulb_s_1':dat_s_}
            else:
                dat_s = self.strong_transform(dat)
                return {'idx_ulb':idx, 'x_ulb_w':dat, 'x_ulb_s': dat_s}


    def __len__(self):
        return len(self.data)



class Network2ImgDataset(Dataset):
    """
    BasicDataset returns a pair of payload_byte network data and labels (targets).
    If targets are not given, BasicDataset returns None as the label.
    This class supports strong augmentation for FixMatch, #TODO: AUGMENTATION FOR NETWORK TRAFFIC DATA?
    and return both weakly and strongly augmented images.
    """

    def __init__(self,
                 alg,
                 data,
                 targets=None,
                 num_classes=None,
                 is_ulb=False,
                 onehot=False,
                 transform=None,
                 strong_transform=None,
                 custom_aug=False,
                 *args, 
                 **kwargs):
        """
        Args:
            alg (_type_): _description_
            data (numpy.array): x_data
            targets (numpy.array, optional): y_data. If not exist, defaults to None.
            num_classes (int, optional): number of label classes. Defaults to None.
            is_ulb (bool, optional): Whether data is unlabeled. Defaults to False.
            onehot (bool, optional): If True, label is converted into onehot vector. Defaults to False.
        """        
        super(Network2ImgDataset, self).__init__()
        self.alg = alg
        self.data = data
        self.targets = targets

        self.num_classes = num_classes
        self.is_ulb = is_ulb
        self.onehot = onehot
        self.customaug_flag = custom_aug
        self.customaug_w = NetImgAugmentation(p1=0.15, p2=0.15)
        self.customaug_s = NetImgAugmentation(p1=0.4, p2=0.4)
        self.transform = transform
        self.strong_transform = strong_transform
        if self.strong_transform is None:
            if self.is_ulb:
                assert self.alg not in ['fullysupervised', 'supervised', 'pseudolabel', 'vat', 'pimodel', 'meanteacher', 'mixmatch'], f"alg {self.alg} requires strong augmentation"
    
    def __sample__(self, idx):
        """ dataset specific sample function """
        # set idx-th target
        if self.targets is None:
            target = None
        else:
            target_ = self.targets[idx]
            target = target_ if not self.onehot else get_onehot(self.num_classes, target_)

        # set augmented images
        data = self.data[idx]
        return data, target

    def __getitem__(self, idx):
        """
        If strong augmentation is not used,
            return weak_augment_data, target
        else:
            return weak_augment_data, strong_augment_data, target
        """
        img, target = self.__sample__(idx)

        if self.transform is None:
            return  {'x_lb':  transforms.ToTensor()(img), 'y_lb': target}
        else:
            if not self.is_ulb:
                return {'idx_lb':idx, 'x_lb': self.augment(img, 'weak'), 'y_lb':target} 
            else:
                if self.alg == 'fullysupervised' or self.alg == 'supervised':
                    return {'idx_ulb':idx} 
                elif self.alg == 'pseudolabel' or self.alg == 'vat':
                    return {'idx_ulb':idx, 'x_ulb_w': self.augment(img, 'weak')} 
                elif self.alg == 'pimodel' or self.alg == 'meanteacher' or self.alg == 'mixmatch':
                    # NOTE x_ulb_s here is weak augmentation
                    return {'idx_ulb': idx, 'x_ulb_w': self.augment(img, 'weak'), 'x_ulb_s': self.augment(img, 'weak')}
                elif self.alg == 'remixmatch':
                    rotate_v_list = [0, 90, 180, 270]
                    rotate_v1 = np.random.choice(rotate_v_list, 1).item()
                    img_s1 = self.augment(img, 'strong')
                    img_s1_rot = torchvision.transforms.functional.rotate(img_s1, rotate_v1)
                    img_s2 = self.augment(img, 'strong')
                    return {'idx_ulb': idx, 'x_ulb_w': self.augment(img, 'weak'), 'x_ulb_s_0': img_s1, 'x_ulb_s_1':img_s2, 'x_ulb_s_0_rot':img_s1_rot, 'rot_v':rotate_v_list.index(rotate_v1)}
                elif self.alg == 'comatch':
                    return {'idx_ulb': idx, 'x_ulb_w': self.augment(img, 'weak'), 'x_ulb_s_0': self.augment(img, 'strong'), 'x_ulb_s_1': self.augment(img, 'strong')} 
                else: #THIS ONE FOR SOFTMATCH
                    return {'idx_ulb': idx, 'x_ulb_w': self.augment(img, 'weak'), 'x_ulb_s': self.augment(img, 'strong')} 


    def augment(self, img:np.ndarray, type:str='weak'):
        """Helper function to apply the weak/strong custom augmentation function for network traffic data.

        Args:
            img (np.ndarray): original image.
            type (str, optional): flag for whether augmentation is weak or strong. Defaults to 'weak'.

        Returns:
            img_a (torch.tensor): augmented image in torch.tensor.
        """
        if isinstance(img, np.ndarray):
            if type == 'weak':   
                img_a = self.customaug_w(img)
                return self.transform(Image.fromarray(img_a.astype(np.uint8)))
            if type == 'strong': 
                img_a = self.customaug_s(img)
                return self.strong_transform(Image.fromarray(img_a.astype(np.uint8)))
        else: raise ValueError('cannot modify non numpy arrays with this function.')


    def __len__(self):
        return len(self.data)