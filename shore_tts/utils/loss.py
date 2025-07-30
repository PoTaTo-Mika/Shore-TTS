import torch
import torch.nn
import torch.nn.functional as F

# 由于L1损失和BCE损失都不会自动忽略mask，所以要做一点措施


def masked_l1_loss(pred, target, mask=None):
    """
    计算 L1 损失，自动忽略 padding 内容。
    参数:
        pred (torch.Tensor): 预测值张量。
        target (torch.Tensor): 目标值张量。
        mask (torch.Tensor, optional): 掩码张量，1 表示有效数据，0 表示 padding。默认为 None。
    返回:
        torch.Tensor: L1 损失值。
    """
    loss = F.l1_loss(pred, target, reduction='none')
    if mask is not None:
        loss = loss * mask
        return loss.sum() / mask.sum() if mask.sum() > 0 else loss.mean()
    return loss.mean()


def masked_bce_loss(pred, target, mask=None):
    """
    计算 BCE 损失，自动忽略 padding 内容。
    参数:
        pred (torch.Tensor): 预测值张量。
        target (torch.Tensor): 目标值张量。
        mask (torch.Tensor, optional): 掩码张量，1 表示有效数据，0 表示 padding。默认为 None。
    返回:
        torch.Tensor: BCE 损失值。
    """
    loss = F.binary_cross_entropy(pred, target, reduction='none')
    if mask is not None:
        loss = loss * mask
        return loss.sum() / mask.sum() if mask.sum() > 0 else loss.mean()
    return loss.mean()

