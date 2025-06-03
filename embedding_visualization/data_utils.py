import torch
import numpy as np
import matplotlib.pyplot as plt

def get_artificial_data():

    def generate_and_encode_signal(signal_fn, start_vals):
        ys = []
        for start in start_vals:
            x = torch.linspace(1 * start, 1 * start + 50, 100)
            y = signal_fn(x)
            y = y.unsqueeze(-1)  # Shape: [1, seq_len, 1]
            ys.append(y)
        return torch.stack(ys, dim=0)

    start_vals = [1, 3, 5, 10, 15, 20, 30, 40, 50, 60]

    # Définir les variantes pour chaque type de signal
    sin_variants = [
        lambda x: torch.sin(x),
        lambda x: torch.sin(x / 2),
        lambda x: torch.sin(x * 2)
    ]

    cos_variants = [
        lambda x: torch.cos(x / 5) + 10,
        lambda x: torch.cos(x / 3) + 5,
        lambda x: 2 * torch.cos(x / 6) + 7
    ]

    quad_variants = [
        lambda x: -x ** 2,
        lambda x: -(x - 25) ** 2,
        lambda x: -0.5 * x ** 2 + 10
    ]

    inv_variants = [
        lambda x: 1 / (x + 1),        
        lambda x: 2 / (x + 1),
        lambda x: -1 / (x + 1)
    ]

    const_variants = [
        lambda x: torch.ones_like(x) * 3,
        lambda x: torch.ones_like(x) * -2,
        lambda x: torch.ones_like(x) * 0.5
    ]

    exp_variants = [
        lambda x: torch.exp(-x / 10),
        lambda x: torch.exp(-x / 20),
        lambda x: 2 * torch.exp(-x / 15)
    ]

    triangle_variants = [
        lambda x: 2 * torch.abs((x % 10) - 5) - 5,
        lambda x: torch.abs((x % 20) - 10) - 5,
        lambda x: 4 * torch.abs(((x / 2) % 5) - 2.5) - 5
    ]

    sawtooth_variants = [
        lambda x: (x % 10) - 5,
        lambda x: ((x % 20) - 10) / 2,
        lambda x: (x % 5) - 2.5
    ]

    relu_like_variants = [
        lambda x: torch.nn.functional.relu(x - 25),
        lambda x: torch.nn.functional.relu(x - 15) * 0.5,
        lambda x: torch.nn.functional.relu(x - 35) * 2
    ]

    noise_variants = [
        lambda x: torch.rand_like(x),                      # bruit uniforme [0, 1]
        lambda x: torch.randn_like(x),                     # bruit gaussien
        lambda x: torch.randn_like(x) * 0.5 + 1.0          # gaussien centré en 1
    ]


    # Générer les embeddings pour chaque type
    sin_ys = torch.cat([generate_and_encode_signal(fn, start_vals) for fn in sin_variants], dim=0)
    cos_ys = torch.cat([generate_and_encode_signal(fn, start_vals) for fn in cos_variants], dim=0)
    quad_ys = torch.cat([generate_and_encode_signal(fn, start_vals) for fn in quad_variants], dim=0)
    inv_ys = torch.cat([generate_and_encode_signal(fn, start_vals) for fn in inv_variants], dim=0)
    const_ys = torch.cat([generate_and_encode_signal(fn, start_vals) for fn in const_variants], dim=0)
    exp_ys = torch.cat([generate_and_encode_signal(fn, start_vals) for fn in exp_variants], dim=0)
    triangle_ys = torch.cat([generate_and_encode_signal(fn, start_vals) for fn in triangle_variants], dim=0)
    saw_ys = torch.cat([generate_and_encode_signal(fn, start_vals) for fn in sawtooth_variants], dim=0)
    relu_ys = torch.cat([generate_and_encode_signal(fn, start_vals) for fn in relu_like_variants], dim=0)
    noise_ys = torch.cat([generate_and_encode_signal(fn, start_vals) for fn in noise_variants], dim=0)

    # Empiler et aplatir
    all_signals = torch.cat([
        sin_ys, cos_ys, quad_ys, inv_ys, const_ys,
        exp_ys, triangle_ys, saw_ys, relu_ys, noise_ys
    ], dim=0)

    n_start_points = len(start_vals)

    return all_signals, n_start_points

def get_visu_artificial_signals():

    x = torch.linspace(1, 50, 100)
    signals = [
        torch.sin(x / 2),
        torch.cos(x / 3) + 5,
        -(x - 25) ** 2,
        2 / (x + 1),
        torch.ones_like(x) * -2,
        torch.exp(-x / 20),
        torch.abs((x % 20) - 10) - 5,
        ((x % 20) - 10) / 2,
        torch.nn.functional.relu(x - 15) * 0.5,
        torch.randn_like(x) * 0.5 + 1.0
    ]
    labels = [
        "Sinusoidal",
        "Cosine",
        "Neg Quadratic",
        "Inverse",
        "Constant",
        "Exponential Decay",
        "Triangle Wave",
        "Sawtooth Wave",
        "ReLU-like",
        "Noisy Signal"
    ]
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'orange', 'brown', 'gray', 'black']

    # Two-column layout
    fig, axes = plt.subplots(2, 5, figsize=(20, 5), sharex=True)

    axes = axes.flatten()  # flatten 2D array of axes for easy indexing

    for i, (signal, ax) in enumerate(zip(signals, axes)):
        ax.plot(x, signal, color=colors[i], label=labels[i])
        ax.legend(loc="upper right", frameon=False, fontsize=8)
        ax.axis('off')

    # Hide any extra axes if needed
    for j in range(len(signals), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    return fig



## From https://github.com/moment-timeseries-foundation-model/moment-research, thanks a lot!


from typing import Tuple

import torch
from torch import nn


class SyntheticDataset(nn.Module):
    def __init__(
        self,
        n_samples: int = 1024,
        seq_len: int = 100,
        freq: int = 1,
        freq_range: Tuple[int, int] = (1, 32),
        amplitude_range: Tuple[int, int] = (1, 32),
        trend_range: Tuple[int, int] = (1, 32),
        baseline_range: Tuple[int, int] = (1, 32),
        noise_mean: float = 0.0,
        noise_std: float = 0.1,
        random_seed: int = 13,
        **kwargs,
    ):
        super(SyntheticDataset, self).__init__()

        self.n_samples = n_samples
        self.seq_len = seq_len
        self.freq = freq
        self.freq_range = freq_range
        self.amplitude_range = amplitude_range
        self.trend_range = trend_range
        self.baseline_range = baseline_range
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.random_seed = random_seed

    def __repr__(self):
        return (
            f"SyntheticDataset(n_samples={self.n_samples},"
            + f"seq_len={self.seq_len},"
            + f"freq={self.freq},"
            + f"freq_range={self.freq_range},"
            + f"amplitude_range={self.amplitude_range},"
            + f"trend_range={self.trend_range},"
            + f"baseline_range={self.baseline_range},"
            + f"noise_mean={self.noise_mean},"
            + f"noise_std={self.noise_std},"
            + f"random_seed={self.random_seed})"
        )

    def _generate_noise(self):
        epsilon = torch.normal(
            mean=self.noise_mean,
            std=self.noise_std,
            size=(self.n_samples, self.seq_len),
        )

        return epsilon

    def _generate_x(self):
        t = (
            torch.linspace(start=0, end=1, steps=self.seq_len)
            .unsqueeze(0)
            .repeat(self.n_samples, 1)
        )
        x = 2 * self.freq * torch.pi * t
        return x, t

    def gen_sinusoids_with_varying_freq(self):
        c = (
            torch.linspace(
                start=self.freq_range[0], end=self.freq_range[1], steps=self.n_samples
            )
            .unsqueeze(1)
            .repeat(1, self.seq_len)
        )
        x, _ = self._generate_x()
        epsilon = self._generate_noise()

        y = torch.sin(c * x) + epsilon
        y = y.unsqueeze(1)

        return y, c

    def gen_sinusoids_with_varying_correlation(self):
        c = (
            torch.linspace(start=0, end=2 * np.pi, steps=self.n_samples)
            .unsqueeze(1)
            .repeat(1, self.seq_len)
        )
        x, _ = self._generate_x()
        epsilon = self._generate_noise()

        y = torch.sin(x + c) + epsilon
        y = y.unsqueeze(1)

        return y, c

    def gen_sinusoids_with_varying_amplitude(self):
        c = (
            torch.linspace(
                start=self.amplitude_range[0],
                end=self.amplitude_range[1],
                steps=self.n_samples,
            )
            .unsqueeze(1)
            .repeat(1, self.seq_len)
        )

        x, _ = self._generate_x()
        epsilon = self._generate_noise()

        y = c * torch.sin(x) + epsilon
        y = y.unsqueeze(1)

        return y, c

    def gen_sinusoids_with_varying_trend(self):
        c = (
            torch.linspace(
                start=self.trend_range[0], end=self.trend_range[1], steps=self.n_samples
            )
            .unsqueeze(1)
            .repeat(1, self.seq_len)
        )
        # c = torch.cat((c, c), dim=0)
        # directions = torch.ones(self.n_samples, self.seq_len)
        # directions[self.n_samples//2:, :] = -1
        x, t = self._generate_x()
        epsilon = self._generate_noise()

        y = torch.sin(x) + t**c + epsilon
        y = y.unsqueeze(1)

        return y, c

    def gen_sinusoids_with_varying_baseline(self):
        c = (
            torch.linspace(
                start=self.baseline_range[0],
                end=self.baseline_range[1],
                steps=self.n_samples,
            )
            .unsqueeze(1)
            .repeat(1, self.seq_len)
        )
        x, _ = self._generate_x()
        epsilon = self._generate_noise()

        y = torch.sin(x) + c + epsilon
        y = y.unsqueeze(1)

        return y, c
    

#  ECG5000 Dataset

from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

def load_ecg_data(train_path: str, test_path: str):
    train = np.loadtxt(train_path)
    test = np.loadtxt(test_path)
    concat = np.concatenate((train, test), axis=0)
    return concat[:, 1:], concat[:, 0]

def preprocess_data(data: np.ndarray):

    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)

    original_length = data.shape[1]
    target_length = 100
    interoplate_func = interp1d(
        np.linspace(0, 1, original_length),
        scaled,
        axis=1
    )
    x_new = np.linspace(0, 1, target_length)
    scaled = interoplate_func(x_new)

    return torch.tensor(scaled, dtype=torch.float32)

class ECGDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx, :].unsqueeze(-1)
        y = self.labels[idx]
        return x, y

def extract_embeddings(model, dataloader, device):
    model.to(device)
    signals, labels = [], []

    for x, y in dataloader:
        signals.append(x.to(device))
        labels.append(y)

    signals = torch.cat(signals).numpy()
    labels = torch.cat(labels).numpy()
    return signals, labels

def get_ecg_signals():
    signals, labels = load_ecg_data(
        train_path="data/ECG5000_TRAIN.txt",
        test_path="data/ECG5000_TEST.txt"
    )
    signals = preprocess_data(signals).unsqueeze(-1)
    return signals, labels