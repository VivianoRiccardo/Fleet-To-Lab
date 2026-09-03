import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import RobustScaler

from config import DEVICE

FEATURE_COLS = [
    "feat_w_vel_mean", "feat_w_vel_std", "feat_curr_mean", "feat_curr_std",
    "feat_acc_x_mean", "feat_acc_x_std", "feat_acc_y_mean", "feat_acc_y_std",
    "feat_acc_z_mean", "feat_acc_z_std", "feat_pitch_mean", "feat_roll_mean"
]


class RoverDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        self.y = torch.tensor(y, dtype=torch.long).to(DEVICE)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_raw_12_features(file_list):
    dfs = []
    for f in file_list:
        if not f.endswith(".csv"):
            f += ".csv"
        if os.path.exists(f):
            dfs.append(pd.read_csv(f))
    if not dfs:
        return None, None
    df = pd.concat(dfs, ignore_index=True).dropna()
    return df[FEATURE_COLS].values, df["slip_class"].values


def add_context_flags(X_12, env_type, rover_id):
    num_samples = len(X_12)

    env = np.tile([1, 0] if env_type == 'earth' else [0, 1], (num_samples, 1))

    rover_map = {1: [1, 0, 0], 2: [0, 1, 0], 3: [0, 0, 1]}
    rov = np.tile(rover_map.get(rover_id, [0, 0, 1]), (num_samples, 1))

    return np.hstack((X_12, env, rov))


def scale_per_domain(X):
    scaler = RobustScaler()
    return scaler.fit_transform(X), scaler
