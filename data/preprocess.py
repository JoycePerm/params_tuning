import json
import os


def load_json(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def remake_one_data(data: dict):
    del data["Result"]["WantedEffect"]

    data["Cut_Params"]["RealPower"] = (
        data["Condition_Params"]["LaserPower"]
        * data["Cut_Params"]["LaserDiodeCurrent"]
        / 100
    )
    del data["Cut_Params"]["LaserDiodeCurrent"]
    del data["Cut_Params"]["LaserRatio"]

    if data["Result"]["score"] == 5:
        data["Result"]["hem"] = 0
        data["Result"]["flySlag"] = 0
        data["Result"]["bubbleSlag"] = 0
        data["Result"]["stratification"] = 0
        data["Result"]["roughness"] = 0
        data["Result"]["breach"] = 0
        data["Result"]["fluidSlag"] = 0
        data["Result"]["slag"] = 0
        data["Result"]["scum"] = 0
        data["Result"]["burrs"] = 0
        data["Result"]["overBreach"] = 0
    else:
        del data["Result"]["isDemo"]

    return data


def remake_data(file_path: str):
    data = load_json(file_path)
    data_new = []
    for i in range(len(data)):
        for j in range(len(data[i])):
            data_new.append(remake_one_data(data[i][j]))

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data_new, f, ensure_ascii=False)

    return


def split_data_by_wanted_effect(file_path: str, target_dir: str):
    data = load_json(file_path)
    data_type0 = []
    data_type1 = []
    for i in range(len(data)):
        wanted_effect = data[i]["Condition_Params"]["WantedEffect"]
        if wanted_effect == 0:
            data_type0.append(data[i])
        elif wanted_effect == 1:
            data_type1.append(data[i])

    target_path0 = os.path.join(target_dir, "data_list_type0.json")
    target_path1 = os.path.join(target_dir, "data_list_type1.json")
    with open(target_path0, "w", encoding="utf-8") as f0:
        json.dump(data_type0, f0, ensure_ascii=False)
    with open(target_path1, "w", encoding="utf-8") as f1:
        json.dump(data_type1, f1, ensure_ascii=False)

    return


if __name__ == "__main__":
    file_path = "/home/yangxiaojiang/code/params-tuning/data/data_list.json"
    target_dir = "/home/yangxiaojiang/code/params-tuning/data"

    # remake_data(file_path)
    split_data_by_wanted_effect(file_path, target_dir)
