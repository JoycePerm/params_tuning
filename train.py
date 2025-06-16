import os
import torch
import torch.nn as nn
import argparse
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from net import ContNet


def main(args):
    data = torch.load(
        args.data_path,
        weights_only=False,
    )
    x = data["x"]
    y = data["y"]
    c = data["c"]

    N = len(data["x"])
    radom_idx = torch.randperm(N)
    train_idx = radom_idx[: int(1.0 * N)]
    val_idx = radom_idx[int(1.0 * N) :]

    x_train, x_val = x[train_idx], x[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    c_train, c_val = c[train_idx], c[val_idx]

    x_dim, y_dim, c_dim = x.size(1), y.size(1), c.size(1)
    h_dim = 32
    model = ContNet(x_dim, c_dim, y_dim, h_dim)
    model = model.cuda()
    optim = AdamW(model.parameters(), lr=0.01)

    n_epoch = 10000
    scheduler = CosineAnnealingLR(optim, T_max=n_epoch, eta_min=0.001)

    loss = nn.MSELoss(reduction="sum")

    for e in range(n_epoch):
        model.train()
        optim.zero_grad()

        x_train = x_train.cuda()
        y_train = y_train.cuda()
        c_train = c_train.cuda()

        y_pred = model(x_train, c_train)
        train_loss = loss(y_pred, y_train).mean()

        train_loss.backward()
        optim.step()
        scheduler.step()

        print(f"Epoch {e}, Train Loss: {train_loss.item()}")

    torch.save(model.state_dict(), f"{args.output_dir}/last_model.pt")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    main(args)
