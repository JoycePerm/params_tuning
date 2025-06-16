import torch
from torch.utils import data


class ParamDataset(data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data["x"])

    def __getitem__(self, idx):
        x = torch.tensor(self.data["x"][idx], dtype=torch.float32)
        y = torch.tensor(self.data["y"][idx], dtype=torch.float32)
        cd = torch.tensor(self.data["cd"][idx], dtype=torch.float32)
        cc = torch.tensor(self.data["cc"][idx], dtype=torch.float32)

        return cc, cd, x, y
