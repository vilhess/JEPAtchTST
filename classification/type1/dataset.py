import torch
from torch.utils.data import Dataset

# Map of signal types to class indices
signal_type_to_label = {
    'sin': 0,
    'cos': 1,
    'quad': 2,
    'inv': 3,
    'const': 4,
    'exp': 5,
    'triangle': 6,
    'saw': 7,
    'relu': 8,
    'noise': 9
}

train_start_vals = [1, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
test_start_vals = [65, 70, 75, 80, 85]

class SignalDataset(Dataset):
    def __init__(self, mode='train', size=100):
        self.data = []
        self.labels = []

        if mode == 'train':
            start_vals = train_start_vals
        elif mode == 'test':
            start_vals = test_start_vals
        else:
            raise ValueError("Invalid mode. Use 'train' or 'test'.")

        variant_dict = {
            'sin': [
                lambda x: torch.sin(x),
                lambda x: torch.sin(x / 2),
                lambda x: torch.sin(x * 2)
            ],
            'cos': [
                lambda x: torch.cos(x / 5) + 10,
                lambda x: torch.cos(x / 3) + 5,
                lambda x: 2 * torch.cos(x / 6) + 7
            ],
            'quad': [
                lambda x: -x ** 2,
                lambda x: -(x - 25) ** 2,
                lambda x: -0.5 * x ** 2 + 10
            ],
            'inv': [
                lambda x: 1 / (x + 1),
                lambda x: 2 / (x + 1),
                lambda x: -1 / (x + 1)
            ],
            'const': [
                lambda x: torch.ones_like(x) * 3,
                lambda x: torch.ones_like(x) * -2,
                lambda x: torch.ones_like(x) * 0.5
            ],
            'exp': [
                lambda x: torch.exp(-x / 10),
                lambda x: torch.exp(-x / 20),
                lambda x: 2 * torch.exp(-x / 15)
            ],
            'triangle': [
                lambda x: 2 * torch.abs((x % 10) - 5) - 5,
                lambda x: torch.abs((x % 20) - 10) - 5,
                lambda x: 4 * torch.abs(((x / 2) % 5) - 2.5) - 5
            ],
            'saw': [
                lambda x: (x % 10) - 5,
                lambda x: ((x % 20) - 10) / 2,
                lambda x: (x % 5) - 2.5
            ],
            'relu': [
                lambda x: torch.nn.functional.relu(x - 25),
                lambda x: torch.nn.functional.relu(x - 15) * 0.5,
                lambda x: torch.nn.functional.relu(x - 35) * 2
            ],
            'noise': [
                lambda x: torch.rand_like(x),
                lambda x: torch.randn_like(x),
                lambda x: torch.randn_like(x) * 0.5 + 1.0
            ]
        }

        for signal_type, variants in variant_dict.items():
            label = signal_type_to_label[signal_type]
            for fn in variants:
                ys = self.generate_and_encode_signal(fn, start_vals, size)
                self.data.extend(ys)
                self.labels.extend([label] * ys.size(0))

    def generate_and_encode_signal(self, signal_fn, start_vals, size):
        ys = []
        for start in start_vals:
            x = torch.linspace(1 * start, 1 * start + 50, size)
            y = signal_fn(x)
            y = y.unsqueeze(-1) 
            ys.append(y)
        return torch.stack(ys, dim=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]