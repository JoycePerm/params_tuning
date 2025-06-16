import json
import os
import numpy as np
import torch


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
KEY_MAP = {
    "x": "Cut_Params",
    "y": "Result",
    "cd": "Condition_Params",
    "cc": "Condition_Params",
}


def convert_data_to_pt(path):
    with open(os.path.join(path, "data_list.json"), "r", encoding="utf-8") as f:
        data_json = json.load(f)

    tgt_data_dict = {"x": [], "y": [], "cd": [], "cc": []}
    for flag in range(len(data_json)):
        for key in DATA_KEY.keys():
            tgt_data_dict[key].append([])
            for sub_key in DATA_KEY[key]:
                tgt_data_dict[key][flag].append(data_json[flag][KEY_MAP[key]][sub_key])

    for key in tgt_data_dict.keys():
        tgt_data_dict[key] = torch.tensor(tgt_data_dict[key])

    torch.save(tgt_data_dict, os.path.join(path, "all_data.pt"))


if __name__ == "__main__":
    path = "/home/chenjiayi/code/params-tuning/data"
    # convert_data_to_pt(path)
    data = torch.load(os.path.join(path, "all_data.pt"), weights_only=False)
    print(data["x"].shape, data["y"].shape, data["cd"].shape, data["cc"].shape)
    print(data["x"], data["y"], data["cd"], data["cc"])
