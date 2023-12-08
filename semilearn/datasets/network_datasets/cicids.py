# Edit by Robin Bhoo

import os
import gc
import copy
import json
import random
import math
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from torchvision import transforms

from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation
from semilearn.datasets.network_datasets.datasetbase import BasicDataset, Network2ImgDataset
from semilearn.datasets.network_datasets.utils import get_network_dataset
from semilearn.datasets.utils import split_ssl_data


def get_cicids(args, alg, name, num_labels, num_classes, data_dir='./data', include_lb_to_ulb=True):
    """Returns a labeled training, unlabeled training, and eval dataset from the UNSW-NB15 dataset.

    Args:
        alg (args.algorithm): _description_
        name (str): name of the dataset fed from args. Used to create data_dir.
        num_labels (int): number of labeled samples in the dataset to be used during training.
        num_classes (int): number of total classes in the target (y value).
        data_dir (str, optional): path to the directory where the processed data will be stored. Defaults to './data'.
        include_lb_to_ulb (bool, optional): If True, labeled data is also included in unlabeled data. Defaults to True.

    Returns:
        lb_dset (BasicDataSet): Contains labeled data and targets.
        ulb_dset (BasicDataSet): Contains unlabeled data.
        eval_dset (BasicDataset): Contains the eval data and targets.
    """
    net_img_transform = args.net_img_transform
    data_dir = os.path.join(data_dir, name.lower())
    crop_size = args.img_size
    crop_ratio = args.crop_ratio

    # Get CICIDS data and split into training-testing data.
    X, Y = get_network_dataset(name)
    train_data, test_data, train_targets, test_targets = train_test_split(X, Y, test_size=0.2, random_state=0)

    if net_img_transform:
        print(f'using custom transforms for net_img data')
        transform_weak = transforms.Compose([
            transforms.PILToTensor(),
        ])

        transform_strong = transforms.Compose([
            transforms.PILToTensor(),
        ])

        transform_val = transforms.Compose([
            transforms.PILToTensor(),
        ])
        
    else:
        transform_weak = transforms.Compose([
            transforms.PILToTensor(),
        ])

        transform_strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            RandAugment(3, 5),
            transforms.PILToTensor(),
        ])

        transform_val = transforms.Compose([
            transforms.PILToTensor(),
        ])

    # Further split the training data into labeled and unlabeled data.
    lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(args, train_data, train_targets, num_classes, 
                                                                lb_num_labels=num_labels,
                                                                ulb_num_labels=args.ulb_num_labels,
                                                                lb_imbalance_ratio=args.lb_imb_ratio,
                                                                ulb_imbalance_ratio=args.ulb_imb_ratio,
                                                                include_lb_to_ulb=include_lb_to_ulb)

    # Load all three groups of data into the DataSet class (labeled, unlabeled, and test)
    if name in ['cicids_img', 'cicids_img_balanced', 'cicids_tiny']:
        lb_dset   = Network2ImgDataset(alg, lb_data, lb_targets, num_classes, 
                                       False, False, transform_weak, None)
        ulb_dset  = Network2ImgDataset(alg, ulb_data, ulb_targets, num_classes, 
                                       True, False, transform_weak, transform_strong)
        eval_dset = Network2ImgDataset(alg, test_data, test_targets, num_classes, 
                                       False, False, transform_val, None)
    else:
        lb_dset   = BasicDataset(alg, lb_data, lb_targets, num_classes, False, False)
        ulb_dset  = BasicDataset(alg, ulb_data, ulb_targets, num_classes, True, False)
        eval_dset = BasicDataset(alg, test_data, test_targets, num_classes, False, False)

    return lb_dset, ulb_dset, eval_dset