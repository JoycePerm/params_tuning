import torch.nn as nn
import torch


def fuse_linear_bn(linear, bn):
    """融合 Linear + BatchNorm1d"""
    fused_linear = nn.Linear(linear.in_features, linear.out_features, bias=True)

    bn_std = torch.sqrt(bn.running_var + bn.eps)
    fused_weight = linear.weight * (bn.weight / bn_std).view(-1, 1)
    fused_bias = (linear.bias - bn.running_mean) * bn.weight / bn_std + bn.bias

    fused_linear.weight.data = fused_weight
    fused_linear.bias.data = fused_bias
    return fused_linear


def fuse_model(model):
    """递归融合模型中的所有 Linear/Conv + BN 组合"""
    for name, module in model.named_children():
        if isinstance(module, nn.Sequential):
            # 处理Sequential中的连续Linear/Conv-BN
            new_seq = []
            i = 0
            while i < len(module):
                if i + 1 < len(module):
                    if isinstance(
                        module[i], (nn.Linear, nn.Conv1d, nn.Conv2d)
                    ) and isinstance(module[i + 1], (nn.BatchNorm1d, nn.BatchNorm2d)):
                        fused_layer = fuse_linear_bn(module[i], module[i + 1])
                        new_seq.append(fused_layer)
                        i += 2  # 跳过BN层
                        continue
                new_seq.append(module[i])
                i += 1
            setattr(model, name, nn.Sequential(*new_seq))
        else:
            fuse_model(module)  # 递归处理子模块
    return model
