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

def train_swarm(features, masks, max_epochs, patience, batch_size, device):
    device = get_device()
    lr = 0.01
    start_epoch = 0

    model = RevNet().to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    criterion = adaswarm.nn.CompressionRateLoss()

    train_loader = DataLoader(GraphFeaturesDataset(features[masks["train_indices"]]), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(GraphFeaturesDataset(features[masks["val_indices"]]), batch_size=batch_size, shuffle=True, drop_last=True)

    for epoch in range(max_epochs):
        print(f"\nEpoch: {epoch}")
        ## train
        model.train()
        running_loss = 0
        batch_losses = []
        f_total = 0
        c_total = 0
        for batch_idx, inputs in enumerate(train_loader):
            inputs = inputs.to(device)

            inputs = Variable(inputs).float()
            t1 = time.time()
            outputs = model(inputs).float()
            f_total += time.time() - t1

            t1 = time.time()
            loss = criterion(outputs)
            c_total += time.time() - t1

            optimizer.zero_grad()  # zero the gradients on each pass before the update
            loss.backward()  # backpropagate the loss through the model
            optimizer.step()  # update the gradients w.r.t the loss

            running_loss += (
                loss.item()
            )  # loss.item() contains the loss of entire mini-batch,
            # but divided by the batch size.
            # here we are summing up the losses as we go

            train_loss = running_loss / (batch_idx + 1)
            forward_time = f_total / (batch_idx + 1)
            compression_time = c_total / (batch_idx + 1)
            batch_losses.append(loss.item())

            progress_bar(
                batch_idx,
                len(train_loader),
                f"""Loss: {train_loss:3f}""",
            )
        print(forward_time, compression_time)

        ## val/test
        model.eval()
        test_loss = 0
        running_loss = 0
        batch_losses = []
        with no_grad():
            for batch_idx, inputs in enumerate(val_loader):
                inputs = inputs.to(device)
                inputs = Variable(inputs).float()
                outputs = model(inputs).float()

                loss = criterion(outputs)
                running_loss += loss.item()
                test_loss = running_loss / (batch_idx + 1)
                batch_losses.append(loss.item())

                progress_bar(
                    batch_idx,
                    len(val_loader),
                    f"""Loss: {test_loss:3f}""",
                )

    torch.save(model.state_dict(), "./results/revresnet.pt")

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
    # f_min = torch.min(features)
    # f_max = torch.max(features)
    # features = (features - f_min) / (f_max - f_min)

    train_swarm(features, masks, max_epochs, patience, batch_size, device)


