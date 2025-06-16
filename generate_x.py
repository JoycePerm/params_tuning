import numpy as np
import torch
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import yaml
from net import ContNet
import json
import time


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


def recommend_params(x, y, c, ckpt_path, st, lr=0.01, steps=1):
    x_dim, y_dim, c_dim = x.size(1), y.size(1), c.size(1)
    h_dim = 32
    model = ContNet(x_dim, c_dim, y_dim, h_dim)

    ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
    # ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt)

    model.eval()

    t_x = x.clone().requires_grad_(True)
    optimizer = torch.optim.SGD([t_x], lr=lr)
    loss = torch.nn.MSELoss(reduction="sum")
    for step in range(steps):
        y_pred = model(t_x, c)

        test_loss = loss(y_pred, y).mean()

        optimizer.zero_grad()
        test_loss.backward()
        optimizer.step()
        dx = t_x - x
        dx = dx * st

    print(f"Loss: {test_loss.item():.4f}")

    print(
        "y_pred:",
        "  ".join(f"{i:.4f}" for i in (y_pred.detach() * 5).squeeze().tolist()),
    )

    return dx.detach()


def main():
    with open("config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    wf = config["WF"]
    X = [config[key] for key in DATA_KEY["x"]]
    CC = [config[key] for key in DATA_KEY["cc"]]
    CD = [config[key] for key in DATA_KEY["cd"]]
    # hem, flySlag, bubbleSlag, stratification, roughness, breach, fluidSlag, slag, scum, burrs, overBreach
    Y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    if wf == 0:
        ckpt_path = "/home/chenjiayi/code/params-tuning/log/type0_LN/last_model.pt"
        hyperp_path = "/home/chenjiayi/code/params-tuning/data/type0/hyper-params.json"
    else:
        ckpt_path = "/home/chenjiayi/code/params-tuning/log/type1_LN/last_model.pt"
        hyperp_path = "/home/chenjiayi/code/params-tuning/data/type1/hyper-params.json"

    with open(hyperp_path, "r") as file:
        data = json.load(file)

    x_mean = torch.tensor([data[key]["mean"] for key in DATA_KEY["x"]])
    x_std = torch.tensor([data[key]["std"] for key in DATA_KEY["x"]])
    cc_mean = torch.tensor([data[key]["mean"] for key in DATA_KEY["cc"]])
    cc_std = torch.tensor([data[key]["std"] for key in DATA_KEY["cc"]])

    # st = torch.tensor([30.0, 30.0, 3.0, 0.3, 3.0])
    st = torch.tensor([100, 100, 100, 100, 100])
    x = ((torch.tensor(X) - x_mean) / x_std).unsqueeze(0)
    cc = (torch.tensor(CC) - cc_mean) / cc_std
    cd_0 = torch.nn.functional.one_hot(torch.tensor(CD)[0], num_classes=4)
    cd_1 = torch.nn.functional.one_hot(torch.tensor(CD)[1], num_classes=3)
    c = torch.cat([cc, cd_0, cd_1]).unsqueeze(0)
    y = (torch.tensor(Y) / 5.0).unsqueeze(0)
    start_time = time.time()

    dx = recommend_params(x, y, c, ckpt_path, st)

    end_time = time.time()
    print(f"推理耗时 {end_time - start_time:.4f} seconds")
    dx = dx * x_std
    print(dx[0][1] * 100)
    dx[0][1] = dx[0][1] * 100 / config["LaserPower"]

    print("模型预测：")
    print(
        "调整参数：",
        "切割速度",
        "激光功率占比",
        "焦点",
        "喷嘴高度",
        "气压",
    )
    print("调整量：", "  ".join(f"{i:.4f}" for i in dx.squeeze().tolist()))


if __name__ == "__main__":
    main()
