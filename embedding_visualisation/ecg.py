import sys
sys.path.append("..")  

import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt

from types import SimpleNamespace
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import interp1d

from models.base_model import PatchTrADencoder

# -----------------------------
# Configuration & Model Loading
# -----------------------------
def load_config(path: str):
    with open(path, "r") as f:
        return SimpleNamespace(**yaml.load(f, Loader=yaml.FullLoader))

def load_encoder(config, checkpoint_path: str):
    model = PatchTrADencoder(config)
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()
    return model

# ---------------------
# Dataset Preparation
# ---------------------
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

# ---------------------
# Embedding Extraction
# ---------------------
def extract_embeddings(model, dataloader, device):
    model.to(device)
    embeddings, labels = [], []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            _, encoded = model(x)
            embeddings.append(encoded.cpu())
            labels.append(y)

    embeddings = torch.cat(embeddings).numpy()
    labels = torch.cat(labels).numpy()
    return embeddings.reshape(embeddings.shape[0], -1), labels

# ---------------------
# Dimensionality Reduction & Plotting
# ---------------------
def reduce_and_plot(embeddings, labels, output_path="ecg_embeddings.png"):

    # t-SNE
    tsne = TSNE(n_components=2)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=labels,
        cmap='viridis',
        alpha=0.6
    )
    plt.colorbar(scatter)
    plt.title("t-SNE of ECG5000 embeddings")
    plt.savefig(output_path, dpi=300)
    plt.close()

# ---------------------
# Main Script
# ---------------------
if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_config("../conf/encoder/config_encoder_base.yaml")
    encoder = load_encoder(config, "../checkpoints/ts_jepa_vicreg_base_1.ckpt")

    values, labels = load_ecg_data(
        'data/ECG5000_TRAIN.txt',
        'data/ECG5000_TEST.txt'
    )
    values = preprocess_data(values)

    dataset = ECGDataset(values, labels)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    embeddings, labels = extract_embeddings(encoder, dataloader, DEVICE)
    reduce_and_plot(embeddings, labels)
