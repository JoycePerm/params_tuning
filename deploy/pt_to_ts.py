# import torch
# from net import ContNet

# x_dim, c_dim, y_dim, h_dim = 5, 5, 11, 32
# model = ContNet(x_dim, c_dim, y_dim, h_dim)
# model.eval()

# scripted_model = torch.jit.script(model)
# scripted_model.save("deploy/dynamic_model_scripted.pt")

# input_x = torch.rand(1, 5)
# input_c = torch.rand(1, 5)

# traced_model = torch.jit.trace(model, (input_x, input_c))
# traced_model.save("deploy/static_model_traced.pt")
# print(traced_model.graph)

# deploy/pt_to_ts.py

import torch
from process_bn import fuse_model
import sys

sys.path.append("..")
from net import ContNet


def export_model():
    # 1. 初始化模型
    x_dim, c_dim, y_dim, h_dim = 5, 10, 11, 32
    x_mean = torch.tensor(
        [
            21.947811126708984,
            14326.2626953125,
            16.138383865356445,
            0.39696937799453735,
            0.7555552124977112,
        ]
    )
    x_std = torch.tensor(
        [
            8.64724063873291,
            6253.00830078125,
            3.2964229583740234,
            0.15321767330169678,
            0.23250168561935425,
        ]
    )
    cc_mean = torch.tensor([21636.36328125, 24.919191360473633, 1.6404038667678833])
    cc_std = torch.tensor([7452.82080078125, 12.775191307067871, 0.37602895498275757])

    x = ((torch.tensor([60.0, 30000.0, -2.3, 0.3, 2.5]) - x_mean) / x_std).unsqueeze(0)
    cc = (torch.tensor([40000, 25, 5]) - cc_mean) / cc_std
    cc = cc.unsqueeze(0)
    cd_0 = torch.nn.functional.one_hot(torch.tensor([0]), num_classes=4)
    cd_1 = torch.nn.functional.one_hot(torch.tensor([0]), num_classes=3)
    c = torch.cat([cc, cd_0, cd_1], 1)

    model = ContNet(x_dim, c_dim, y_dim, h_dim)

    # 2. 加载预训练权重（关键！）
    checkpoint = torch.load("../log/type0_LN/last_model.pt", map_location="cpu")
    # print(checkpoint.keys())
    model.load_state_dict(checkpoint)

    # 3. 切换到推理模式
    model.eval()

    # 4. BN吸附
    fused_model = fuse_model(model)

    # 5. 导出TorchScript
    example_inputs = (x, c)
    traced_model = torch.jit.trace(fused_model, example_inputs)
    traced_model.save("contnet_fused_0.pt")

    # 验证
    with torch.no_grad():
        orig_out = model(x, c)
        fused_out = traced_model(*example_inputs)
        print(orig_out, fused_out)
        assert torch.allclose(orig_out, fused_out, atol=1e-5), "吸附后输出不一致！"
    print("导出成功！")


if __name__ == "__main__":
    export_model()
