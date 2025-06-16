import torch


def inverse_normalize_adjustment(delta_x, model, mean_input, std_input):
    # 假设 model 包含 LN 和 BN 层
    # 1. 逆BN（需保存BN的mu和sigma）
    if hasattr(model, "bn"):
        gamma = model.bn.weight
        beta = model.bn.bias
        mu_bn = model.bn.running_mean
        var_bn = model.bn.running_var
        delta_x = delta_x * gamma / torch.sqrt(var_bn + 1e-5)  # 逆BN缩放

    # 2. 逆LN（需保存LN的mu和sigma）
    if hasattr(model, "ln"):
        mu_ln = delta_x.mean(dim=-1, keepdim=True)  # 需前向时保存
        var_ln = delta_x.var(dim=-1, keepdim=True)
        delta_x = delta_x * torch.sqrt(var_ln + 1e-5)  # 逆LN缩放

    # 3. 逆输入归一化
    delta_x_original = delta_x * std_input
    return delta_x_original
