import os
import json
import numpy as np
import torch


def get_mean_std(data):
    x_mean = data["x"].mean(0)
    x_std = data["x"].std(0)
    cc_mean = data["cc"].mean(0)
    cc_std = data["cc"].std(0)

    return x_mean, x_std, cc_mean, cc_std


def save_hyper_params(path):
    data = torch.load(os.path.join(path, "all_data.pt"), weights_only=False)
    x_mean, x_std, cc_mean, cc_std = get_mean_std(data)
    hyper_params = {
        "WorkSpeed": {"mean": x_mean[0].item(), "std": x_std[0].item()},
        "RealPower": {"mean": x_mean[1].item(), "std": x_std[1].item()},
        "Focus": {"mean": x_mean[2].item(), "std": x_std[2].item()},
        "CutHeight": {"mean": x_mean[3].item(), "std": x_std[3].item()},
        "GasPressure": {"mean": x_mean[4].item(), "std": x_std[4].item()},
        "LaserPower": {"mean": cc_mean[0].item(), "std": cc_std[0].item()},
        "Thickness": {"mean": cc_mean[1].item(), "std": cc_std[1].item()},
        "NozzleDiameter": {"mean": cc_mean[2].item(), "std": cc_std[2].item()},
        "GasType": {"range": 3},
        "NozzleName": {"range": 2},
        "Material": {"range": 0},
    }
    with open(os.path.join(path, "all_hyper-params.json"), "w", encoding="utf-8") as f:
        json.dump(hyper_params, f, ensure_ascii=False)

    return


def normalize(path):
    data = torch.load(os.path.join(path, "all_data.pt"), weights_only=False)
    x_mean, x_std, cc_mean, cc_std = get_mean_std(data)
    x = (data["x"] - x_mean.unsqueeze(0)) / x_std.unsqueeze(0)
    cc = (data["cc"] - cc_mean.unsqueeze(0)) / cc_std.unsqueeze(0)
    cd_0 = torch.nn.functional.one_hot(data["cd"][:, 0], num_classes=4)
    cd_1 = torch.nn.functional.one_hot(data["cd"][:, 1], num_classes=3)
    c = torch.cat([cc, cd_0, cd_1], dim=1)
    y = data["y"] / 5.0
    normalized_data = {
        "x": x,
        "y": y,
        "c": c,
    }
    torch.save(normalized_data, os.path.join(path, "all_data_normalized.pt"))


if __name__ == "__main__":
    path = "/home/chenjiayi/code/params-tuning/data"
    # normalize(path)
    save_hyper_params(path)
