import os
import torch
import torch.nn as nn
import argparse
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from net import ContNet
import matplotlib.pyplot as plt
import sys

sys.path.append("..")
import numpy as np


def compute_gradient_saliency(model, x, c, y, target_output_dim=0):
    """计算输入 (x, c, y) 对模型输出的梯度显著性（贡献度）"""
    # 确保输入需要梯度
    x.requires_grad_(True)
    c.requires_grad_(True)
    y.requires_grad_(True)

    # 前向传播
    dx_pred = model(x, c, y)

    # 选择目标输出维度（默认第0维）
    target = dx_pred[:, target_output_dim].sum()  # 求和以保持梯度形状

    # 反向传播
    target.backward()

    # 获取梯度并取绝对值
    x_grad = x.grad.abs().mean(dim=0)  # 沿批次维度平均
    c_grad = c.grad.abs().mean(dim=0)
    y_grad = y.grad.abs().mean(dim=0)

    return {
        "x_importance": x_grad.detach().cpu().numpy(),
        "c_importance": c_grad.detach().cpu().numpy(),
        "y_importance": y_grad.detach().cpu().numpy(),
    }


def main(args):
    data = torch.load(
        args.data_path,
        weights_only=False,
    )
    x = data["x"]
    y = data["y"]
    c = data["c"]
    dx = data["dx"]

    N = len(data["x"])
    radom_idx = torch.randperm(N)
    train_idx = radom_idx[: int(1.0 * N)]
    val_idx = radom_idx[int(1.0 * N) :]

    x_train, x_val = x[train_idx], x[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    c_train, c_val = c[train_idx], c[val_idx]
    dx_train, dx_val = dx[train_idx], dx[val_idx]

    x_dim, y_dim, c_dim, dx_dim = x.size(1), y.size(1), c.size(1), dx.size(1)
    h_dim = 32
    model = ContNet(x_dim, c_dim, y_dim, dx_dim, h_dim)
    model = model.cuda()
    optim = AdamW(model.parameters(), lr=0.01)

    n_epoch = 5001
    scheduler = CosineAnnealingLR(optim, T_max=n_epoch, eta_min=0.001)

    loss = nn.MSELoss(reduction="sum")

    for e in range(n_epoch):
        model.train()
        optim.zero_grad()

        x_train = x_train.cuda()
        y_train = y_train.cuda()
        c_train = c_train.cuda()
        dx_train = dx_train.cuda()

        dx_pred = model(x_train, c_train, y_train)
        train_loss = loss(dx_pred, dx_train).mean()

        train_loss.backward()
        optim.step()
        scheduler.step()

        print(f"Epoch {e}, Train Loss: {train_loss.item()}")

    torch.save(model.state_dict(), f"{args.output_dir}/last_model.pt")

    model.eval()
    for i in range(dx_dim):
        saliency = compute_gradient_saliency(
            model,
            x_train.clone(),  # 用少量样本计算
            c_train.clone(),
            y_train.clone(),
            target_output_dim=i,  # 分析对第0维输出的贡献
        )
        print(saliency)

        # Combine all importance scores
        x_imp = saliency["x_importance"]
        c_imp = saliency["c_importance"]
        y_imp = saliency["y_importance"]

        # Create labels for the plot
        labels = []
        labels.extend([f"x_{j}" for j in range(len(x_imp))])
        labels.extend([f"c_{j}" for j in range(len(c_imp))])
        labels.extend([f"y_{j}" for j in range(len(y_imp))])

        # Combine data
        data = np.concatenate((x_imp, c_imp, y_imp))

        plt.figure(figsize=(12, 6))
        plt.bar(range(len(data)), data)
        plt.title(f"Input Feature Importance for dx[{i}]")
        plt.xticks(range(len(data)), labels, rotation=90)
        plt.ylabel("Gradient Saliency")
        plt.tight_layout()  # Prevent label cutoff

        # Save and show
        plt.savefig(f"dx_{i}_importance.png", bbox_inches="tight", dpi=300)
        plt.show()
        plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    main(args)

