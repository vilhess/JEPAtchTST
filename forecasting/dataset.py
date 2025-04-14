import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class ETTDataset(Dataset):
    def __init__(self, path, seq_len=100, target_len=100, train=True, univariate=False, target='OT'):

        self.seq_len = seq_len
        self.target_len = target_len
        self.train = train

        self.data = pd.read_csv(path)
        self.data.drop(columns=['date'], inplace=True)

        if univariate:
            self.data = self.data[[target]]

        train_size = int(len(self.data) * 0.8)
        train = self.data[:train_size].values

        scaler = StandardScaler()
        scaler.fit(train)

        self.data = scaler.transform(self.data.values)

        if self.train:
            self.data = self.data[:train_size]
        else:
            self.data = self.data[train_size - seq_len:]

    def __len__(self):
        return len(self.data) - self.seq_len - self.target_len
    
    def __getitem__(self, index):
        x = self.data[index:index + self.seq_len]
        y = self.data[index + self.seq_len:index + self.seq_len + self.target_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)