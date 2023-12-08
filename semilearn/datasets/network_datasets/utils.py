# Edit by Robin Bhoo
"""
Collection of helper functions for generating DataSet object specific to network traffic data.
"""
import os
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import sampler, DataLoader
import torch.distributed as dist
from io import BytesIO


# Is this a good way to manage dataset paths?
NETWORK_DATASET_DICT = {
    'unsw'       : "/home/jovyan/wire/DataSets/PayloadByte_UNSW/Payload_data_UNSW.csv",
    'cicids'     : "/home/jovyan/wire/DataSets/PayloadByte_CICIDS17/Payload_data_CICIDS2017.csv",
    'unsw_img'   : "/home/jovyan/data/network2img_dataset/unsw/unsw_full.npz",
    'cicids_img' : "/home/jovyan/data/network2img_dataset/cicids/cicids_full.npz",
    'cicids_img_balanced' : "/home/jovyan/data/network2img_dataset/cicids/cicids_balanced_compressedx2.npz",
    'unsw_img_balanced'   : "/home/jovyan/data/network2img_dataset/unsw/unsw_balanced_compressedx2.npz",
    'cicids_tiny': "/home/jovyan/data/network2img_dataset/cicids/cicids_tiny.npz",
    'unsw_tiny': "/home/jovyan/data/network2img_dataset/unsw/unsw_tiny.npz",
}

def get_network_dataset(name:str):
    """Given the name of the dataset, find the path to the dataset and conduct the following:
    1. Read dataset into pd.DataFrame and map all categorical features into numeric values.
    2. Create two separate objects for x (payload) and y (traffic type) values.

    Args:
        name (str): name of the dataset. This is used to look up the path to the dataset in NET_DSET_DICT.
    """
    # Step 1,
    if name in ['cicids_img', 'unsw_img', 'cicids_img_balanced', 'unsw_img_balanced', 'unsw_tiny', 'cicids_tiny']:
        dset = np.load(NETWORK_DATASET_DICT[name.lower()], allow_pickle=True)
    elif name == 'unsw' or name == 'cicids':
        dset = pd.read_csv(NETWORK_DATASET_DICT[name.lower()])
    else:
        raise NotImplementedError(f'get_network_dataset not implemented for {name}.')

    PROTOCOL_MAP = {val: idx for idx, val in enumerate(np.unique(dset['protocol']))}
    LABEL_MAP = {val: idx for idx, val in enumerate(np.unique(dset['label']))}
    if isinstance(dset, np.lib.npyio.NpzFile): 
        dset = {'label': dset['label'], 
                'net_img': dset['net_img'],
                'protocol': dset['protocol'],
                }
    dset['protocol'] = [PROTOCOL_MAP[val] for val in dset['protocol']]
    dset['label'] = [LABEL_MAP[val] for val in dset['label']]

    # Step 2
    if name not in ['cicids_img', 'unsw_img', 'cicids_img_balanced', 'unsw_img_balanced', 'unsw_tiny', 'cicids_tiny']:
        x, y = dset.loc[:, dset.columns!='label'].to_numpy(dtype=np.float32), dset['label'].to_numpy(dtype=int)
    else:
        x, y = dset['net_img'], dset['label']
    
    return x, y
