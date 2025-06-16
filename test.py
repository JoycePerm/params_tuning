import os
import torch
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

from net import ContNet


DATA_KEY = {
    "x": ["WorkSpeed", "RealPower", "Focus", "CutHeight", "GasPressure"],
    "y": [
        "hem",
        "flySlag",
        "bubbleSlag",
        "stratification",
        "roughness",
        "breach",
        "fluidSlag",
        "slag",
        "scum",
        "burrs",
        "overBreach",
    ],
    "cd": ["GasType", "NozzleName"],
    "cc": ["LaserPower", "Thickness", "NozzleDiameter"],
}


def main(path):
    data = torch.load(
        "/home/chenjiayi/code/params-tuning/data/type0/data_normalized.pt",
        weights_only=False,
    )
    x = data["x"]
    y = data["y"]
    c = data["c"]

    x_dim, y_dim, c_dim = x.size(1), y.size(1), c.size(1)
    h_dim = 32
    model = ContNet(x_dim, c_dim, y_dim, h_dim)

    ckpt_path = os.path.join(path, "last_model.pt")
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt)

    model.eval()

    with torch.no_grad():
        y_pred = model(x, c)

    y_pred = y_pred.cpu().numpy()
    y = y.cpu().numpy()
    for dim in range(y_dim):
        r2 = r2_score(y[:, dim], y_pred[:, dim])
        mae = mean_absolute_error(y[:, dim], y_pred[:, dim])

        plt.figure(figsize=(8, 6))
        plt.scatter(y[:, dim], y_pred[:, dim], alpha=0.5)
        plt.plot(
            [0.0, 1.0],
            [0.0, 1.0],
            "r--",
            label="Perfect Prediction",
        )
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        plt.title(f"{DATA_KEY['y'][dim]}- R^2: {r2:.4f}, MAE: {mae:.4f}")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.legend()
        plt.savefig(os.path.join(path, f"{DATA_KEY['y'][dim]}.png"))


if __name__ == "__main__":
    path = "/home/chenjiayi/code/params-tuning/log/type0_LN"
    main(path)
