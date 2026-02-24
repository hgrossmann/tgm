import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

import pandas as pd

data_raw = pd.read_csv("C:/Users/Startklar/Documents/Uni/MA/functional_flow_matching/data/aemet.csv",index_col=0)
data = torch.tensor(data_raw.values).squeeze().unsqueeze(1)
x_grid = torch.linspace(0,1,365)

data = data.permute(0, 2, 1).contiguous()

N = len(data)
n_train = int(0.8 * N)
n_val = N - n_train
train_x, val_x = random_split(
    TensorDataset(data),
    [n_train, n_val],
    generator=torch.Generator().manual_seed(123)
)

train_t = x_grid.repeat(n_train, 1)
val_t = x_grid.repeat(n_val, 1)

#train_ds = aemetDataset(train, x_grid)
#val_ds = aemetDataset(val, x_grid)

train_loader = DataLoader(train_x, batch_size=73, shuffle=True)
val_loader   = DataLoader(val_x,   batch_size=73, shuffle=False)

# check
x_train = next(iter(train_loader))[0]
x_val   = next(iter(val_loader))[0]

print(train_x)