import os
import sys
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import sar
from models.revnet import RevNet

from torch.autograd.grad_mode import no_grad
from torch.autograd import Variable

sys.path.append(os.path.join(os.getcwd(), "AdaSwarm"))


# pylint: disable=C0411, E0401, C0413
import adaswarm.nn 
from adaswarm.utils import progress_bar
from adaswarm.data import DataLoaderFetcher
from adaswarm.utils.options import (
    number_of_epochs,
    get_device,
    log_level,
)

import fpzip

class GraphFeaturesDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        f = self.features[idx,:]
        return f[None,:]

if __name__ == '__main__':
    partitioning_json_file = "./partition_data_2/ogbn-arxiv.json"
    rank = 0
    device = "cpu"
    max_epochs = 10
    patience=10
    batch_size = 100 

    # Load DGL partition data
    partition_data = sar.load_dgl_partition_data(partitioning_json_file, rank, False, device)
    masks = {}
    for mask_name, indices_name in zip(['train_mask', 'val_mask', 'test_mask'],
                    ['train_indices', 'val_indices', 'test_indices']):
        boolean_mask = sar.suffix_key_lookup(partition_data.node_features,
                            mask_name)
        masks[indices_name] = boolean_mask.nonzero(
            as_tuple=False).view(-1).to(device)
    labels = sar.suffix_key_lookup(partition_data.node_features,
                    'labels').long().to(device)
    features = sar.suffix_key_lookup(
            partition_data.node_features, 'features').to(device)
    
    train_loader = DataLoader(GraphFeaturesDataset(features[masks["train_indices"]]), batch_size=batch_size, shuffle=True, drop_last=True)
    
    device = get_device()
    model = RevNet().to(device)
    state = torch.load("./results/revresnet.pt")
    model.load_state_dict(state)

    for batch_idx, inputs in enumerate(train_loader):
        inputs = inputs.to(device)
        inputs = Variable(inputs).float()
        outputs = model(inputs).float()
        outputs = outputs.cpu().detach().numpy()
        compressed_bytes = fpzip.compress(outputs, precision=0, order='C')
        decompressed_outputs = fpzip.decompress(compressed_bytes, order='C')
        if decompressed_outputs.shape[0] == 1:
            decompressed_outputs = decompressed_outputs[0]
        decompressed_outputs = Variable(torch.Tensor(decompressed_outputs)).float()
        decompressed_inputs = model.inverse(decompressed_outputs) 
        print(inputs[0], decompressed_inputs[0])
