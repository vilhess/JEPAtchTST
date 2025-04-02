import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.preprocessing import StandardScaler

class UniDataset(Dataset):
    def __init__(self, values, ws, scaler=False):
        """
        Dataset personnalisé pour gérer les données temporelles avec une fenêtre glissante.

        Args:
            values (numpy.array): Les valeurs des données.
            ws (int): La taille de la fenêtre glissante.
            scaler (bool): Si True, applique la mise à l'échelle des données avec StandardScaler.
        """
        super().__init__()
        self.values = values
        self.ws = ws
        if scaler:
            self.scaler = StandardScaler()
            self.values = self.scaler.fit_transform(self.values)
        self.values = torch.tensor(self.values, dtype=torch.float32)

    def __len__(self):
        return len(self.values) - self.ws + 1

    def __getitem__(self, idx):
        return self.values[idx:idx + self.ws]

def create_dataloader(ws, data_dir='data/raw', batch_size=64, collator=None):
    """
    Fonction pour créer un DataLoader combinant des datasets réels et synthétiques.

    Args:
        ws (int): Taille de la fenêtre glissante.
        data_dir (str): Répertoire contenant les fichiers CSV des datasets réels.
        batch_size (int): Taille des lots pour le DataLoader.

    Returns:
        DataLoader: Un DataLoader combiné.
    """
    all_tables = []

    # Charger les datasets réels depuis les fichiers CSV
    for data_name in os.listdir(data_dir):
        df = pd.read_csv(f'{data_dir}/{data_name}')
        columns = df.columns
        for col in columns:
            values = df[col].values
            values = values.reshape(-1, 1)
            dataset = UniDataset(values, ws, scaler=True)
            all_tables.append(dataset)

    # Créer des datasets synthétiques
    x = np.linspace(1, 100, 1000)
    y1 = np.sin(x) + np.random.normal(0, 0.5, 1000)
    y2 = np.sin(2*x) + np.random.normal(0, 0.5, 1000)
    y3 = np.sin(3*x) + np.random.normal(0, 0.5, 1000)
    y4 = np.cos(x) + np.random.normal(0, 0.5, 1000)
    y5 = np.cos(2*x) + np.random.normal(0, 0.5, 1000)
    y6 = np.cos(3*x) + np.random.normal(0, 0.5, 1000)
    y7 = np.tanh(x) + np.random.normal(0, 0.5, 1000)
    y8 = np.tanh(2*x) + np.random.normal(0, 0.5, 1000)
    y9 = np.tanh(3*x) + np.random.normal(0, 0.5, 1000)
    y10 = y1 + y4 + y7
    y11 = y2 + y5 + y8
    y12 = y3 + y6 + y9
    y13 = y1 + y2 + y3
    y14 = y4 + y5 + y6
    y15 = y7 + y8 + y9
    y16 = y1 + y5 + y9
    y17 = y2 + y6 + y7
    y18 = y3 + y4 + y8

    # Ajouter les datasets synthétiques
    y = np.array([y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17, y18])
    for yy in y:
        all_tables.append(UniDataset(yy.reshape(-1, 1), ws, scaler=False))

    # Combiner tous les datasets
    combined_dataset = ConcatDataset(all_tables)

    # Créer le DataLoader
    dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, num_workers=21, collate_fn=collator)
    
    return dataloader

if __name__ == '__main__':
    ws = 100  # Taille de la fenêtre glissante
    batch_size = 64
    dataloader = create_dataloader(ws, batch_size=batch_size)

    # Vérification de la forme des lots
    for batch in dataloader:
        print(batch.shape)
        break  # Afficher un seul batch pour vérifier la forme
