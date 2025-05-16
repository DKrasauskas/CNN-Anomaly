import numpy as np
import torch
from torch.utils.data import Subset, ConcatDataset
from torch.utils.data import random_split
from src.data.Element import VoltageDataset
from logs.log import Log
import src.network.Network as net
def get_dataset(path, validation_size, label_path, remove_duplicates = False, info = False):
    """
    Function to get the dataset from the path
    Parameters
    ----------
    path : str
        Path to the dataset.
    Returns
    -------
    dataset : Dataset
        Dataset object.
    """
    paths = open(path, "r").read().split('\n')[:-1]
    set = VoltageDataset(paths, label_path)
    train_size = len(set.dataset) - validation_size
    train_dataset, val_dataset = random_split(set.dataset, [train_size, validation_size])
    if remove_duplicates:
        rem_ind = []
        for i, (input, target) in enumerate(val_dataset):
            for j, (input2, target2) in enumerate(train_dataset):
                if abs(target.item() - target2.item()) < 0.005:
                    rem_ind.append(j)
        indices = np.array(range(len(train_dataset)))
        keep_indices = indices[~np.isin(indices, rem_ind)]
        move_indices = indices[np.isin(indices, rem_ind)]
        moved_subset = torch.utils.data.Subset(train_dataset, move_indices)
        train_dataset = torch.utils.data.Subset(train_dataset, keep_indices)

        if isinstance(val_dataset, ConcatDataset):
            val_dataset = ConcatDataset([val_dataset, moved_subset])
        else:
            val_dataset = ConcatDataset([val_dataset, moved_subset])
        
    return train_dataset, val_dataset
