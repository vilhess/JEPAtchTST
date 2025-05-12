import os
import pandas as pd
import torch
from torch.utils.data import Dataset, ConcatDataset
from sklearn.preprocessing import StandardScaler

class TSDataset(Dataset):
    def __init__(self, path, seq_len=100, mode="train", univariate=False, target='OT', class_idx=0):
        self.seq_len = seq_len
        self.mode = mode
        self.class_idx = class_idx

        self.data = pd.read_csv(path)
        self.data.drop(columns=['date'], inplace=True)

        if univariate:
            self.data = self.data[[target]]

        train_size = int(len(self.data) * 0.7)
        test_size = int(len(self.data) * 0.2)
        val_size = len(self.data) - train_size - test_size

        train = self.data[:train_size].values

        scaler = StandardScaler()
        scaler.fit(train)

        self.data = scaler.transform(self.data.values)

        if mode=="train":
            self.data = self.data[:train_size]
        elif mode == "val":
            self.data = self.data[train_size-seq_len:train_size + val_size]
        else:
            self.data = self.data[train_size + val_size - seq_len:]

    def __len__(self):
        return len(self.data) - self.seq_len + 1
    
    def __getitem__(self, index):
        x = self.data[index:index + self.seq_len]
        return torch.tensor(x, dtype=torch.float32), self.class_idx

def load_concat_datasets(seq_len=100, base_path='../../forecasting/data/', univariate=True, target='OT'):
    all_datasets = os.listdir(base_path)
    all_datasets = [dataset for dataset in all_datasets if dataset.endswith('.csv')]
    train_datasets, val_datasets, test_datasets = [], [], []
    signal_type_to_label = {}

    for class_idx, dataset in enumerate(all_datasets):
        dataset_path = os.path.join(base_path, dataset)
        dataset_name = dataset.split('.')[0]
        signal_type_to_label[dataset_name] = class_idx

        train = TSDataset(
            path=dataset_path,
            seq_len=seq_len,
            mode="train",
            univariate=univariate,
            target=target,
            class_idx=class_idx
        )
        valset = TSDataset(
            path=dataset_path,
            seq_len=seq_len,
            mode="val",
            univariate=univariate,
            target=target,
            class_idx=class_idx
        )
        test = TSDataset(
            path=dataset_path,
            seq_len=seq_len,
            mode="test",
            univariate=univariate,
            target=target,
            class_idx=class_idx
        )

        train_datasets.append(train)
        val_datasets.append(valset)
        test_datasets.append(test)

    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)
    test_dataset = ConcatDataset(test_datasets)

    return train_dataset, val_dataset, test_dataset, signal_type_to_label
