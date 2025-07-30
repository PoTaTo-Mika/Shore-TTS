import torch.optim as optim
import torch.nn as nn

def get_optimizer(optimizer_name, params, **kwargs):
    # 移除 'name' 键，避免传递给优化器构造函数
    optimizer_kwargs = {k: v for k, v in kwargs.items() if k != 'name'}
    
    if optimizer_name == "AdamW":
        return optim.AdamW(params, **optimizer_kwargs)
    elif optimizer_name == "Adam":
        return optim.Adam(params, **optimizer_kwargs)
    elif optimizer_name == "SGD":
        return optim.SGD(params, **optimizer_kwargs)
    else:
        raise ValueError(f"不支持的优化器: {optimizer_name}")

def get_loss_function(loss_name, **kwargs):
    # 移除 'name' 相关的键，避免传递给损失函数构造函数
    loss_kwargs = {k: v for k, v in kwargs.items() if not k.endswith('_type') and not k.endswith('_weight') and k != 'name'}
    
    if loss_name == "l1":
        return nn.L1Loss(**loss_kwargs)
    elif loss_name == "l2":
        return nn.MSELoss(**loss_kwargs)
    elif loss_name == "BCE":
        return nn.BCELoss(**loss_kwargs)
    elif loss_name == "BCEWithLogitsLoss":
        return nn.BCEWithLogitsLoss(**loss_kwargs)
    elif loss_name == "CrossEntropyLoss":
        return nn.CrossEntropyLoss(**loss_kwargs)
    elif loss_name == "NLLLoss":
        return nn.NLLLoss(**loss_kwargs)
    else:
        raise ValueError(f"不支持的损失函数: {loss_name}")

def get_scheduler(scheduler_name, optimizer, **kwargs):
    if scheduler_name == "StepLR":
        return optim.lr_scheduler.StepLR(optimizer, **kwargs)
    elif scheduler_name == "CosineAnnealingLR":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    elif scheduler_name == "ExponentialLR":
        return optim.lr_scheduler.ExponentialLR(optimizer, **kwargs)
    else:
        raise ValueError(f"不支持的调度器: {scheduler_name}")