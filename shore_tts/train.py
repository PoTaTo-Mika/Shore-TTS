import torch
import torch.nn as nn
import torch.nn.functional as F
# 处理
import logging
import json
import os
import time
import sys

# 设置根目录
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

# 内部包
from shore_tts.models.tacotron import Tacotron1 # 直接调用模型族
from shore_tts.datasets import shore_dataset 
from shore_tts.utils.get_func import get_optimizer, get_loss_function, get_scheduler
from shore_tts.utils.loss import masked_l1_loss, masked_bce_loss
# 分布式训练
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import argparse
import subprocess
import random
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import numpy as np
# 我们从头开始就用分布式训练的方案，单卡训练是分布式训练中的一种特殊情况

os.makedirs("logs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/train.log')
    ]
)

with open("shore_tts/configs/tacotron.json", "r", encoding='utf-8') as f:
    config = json.load(f)

logging.info(f"✓ 成功加载配置文件: {config}")
# 训练配置
training_config = config['training']
data_dir = training_config['data_dir']
device = training_config['device']
batch_size = training_config['batch_size']
epochs = training_config['epochs']
learning_rate = training_config['learning_rate']
weight_decay = training_config['weight_decay']
grad_clip = training_config['grad_clip']
save_every_steps = training_config['save_every_steps']
save_every_epoch = training_config['save_every_epoch']
log_every_steps = training_config['log_every_steps']
num_workers = training_config['num_workers']
optimizer_config = training_config['optimizer']
scheduler_config = training_config['scheduler']
loss_config = training_config['loss']

# 评估配置
eval_config = config['evaluation']
eval_every_steps = eval_config['eval_every_steps']

# 模型配置
model_config = config['model']

# 使用配置文件中的参数初始化模型
model = Tacotron1(
    phoneme_vocab_size=model_config['phoneme_vocab_size'],
    embedding_dim=model_config['embedding_dim'],
    encoder_prenet_hidden_dim=model_config['encoder_prenet_hidden_dim'],
    encoder_prenet_output_dim=model_config['encoder_prenet_output_dim'],
    cbhg_k=model_config['cbhg_k'],
    cbhg_conv_channels=model_config['cbhg_conv_channels'],
    cbhg_highway_layers=model_config['cbhg_highway_layers'],
    attention_dim=model_config['attention_dim'],
    decoder_prenet_hidden_dim=model_config['decoder_prenet_hidden_dim'],
    decoder_prenet_output_dim=model_config['decoder_prenet_output_dim'],
    decoder_rnn_dim=model_config['decoder_rnn_dim'],
    num_mels=model_config['num_mels'],
    max_decoder_steps=model_config['max_decoder_steps']
)

# 数据集配置
dataset_config = config['dataset']

# 创建完整数据集
full_dataset = shore_dataset.ShoreTTSDataset(
    data_root=data_dir,
    text_subdir=dataset_config['text_subdir'],
    mel_subdir=dataset_config['mel_subdir'],
    pad_token_id=dataset_config['pad_token_id'],
    eos_token_id=dataset_config['eos_token_id'],
    add_eos=dataset_config['add_eos']
)

# 数据集分割 - 训练集和验证集
dataset_size = len(full_dataset)
eval_size = int(eval_config['split_rate'] * dataset_size)
train_size = dataset_size - eval_size

logging.info(f"数据集统计:")
logging.info(f"  - 总数据量: {dataset_size}")
logging.info(f"  - 训练集大小: {train_size}")
logging.info(f"  - 验证集大小: {eval_size}")

# 使用随机分割
train_dataset, eval_dataset = torch.utils.data.random_split(
    full_dataset, 
    [train_size, eval_size],
    generator=torch.Generator().manual_seed(42)  # 设置随机种子确保可重现
)

# 创建DataLoader
from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    collate_fn=shore_dataset.collate_fn,
    pin_memory=True if device == "cuda" else False,
    drop_last=True  # 训练时丢弃最后不完整的批次
)

eval_dataloader = DataLoader(
    dataset=eval_dataset,
    batch_size=eval_config['batch_size'],
    shuffle=False,  # 验证时不需要打乱
    num_workers=num_workers,
    collate_fn=shore_dataset.collate_fn,
    pin_memory=True if device == "cuda" else False,
    drop_last=False  # 验证时保留所有数据
)

logging.info(f"✓ 数据加载器创建完成:")
logging.info(f"  - 训练批次数: {len(train_dataloader)}")
logging.info(f"  - 验证批次数: {len(eval_dataloader)}")
logging.info(f"  - 训练批大小: {batch_size}")
logging.info(f"  - 验证批大小: {eval_config['batch_size']}") 

# 优化器
optimizer = get_optimizer(
    optimizer_config['name'], 
    model.parameters(), 
    **optimizer_config
    )
# 损失函数
mel_loss = masked_l1_loss
gate_loss = masked_bce_loss

scheduler = get_scheduler(
    scheduler_config['name'],
    optimizer,
    **{k: v for k, v in scheduler_config.items() if k != 'name'}
)

def setup_distributed(rank, world_size):
    """初始化分布式训练环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """清理分布式训练环境"""
    dist.destroy_process_group()

def create_mask_from_lengths(lengths, max_len):
    """根据实际长度创建掩码"""
    batch_size = lengths.size(0)
    mask = torch.arange(max_len, device=lengths.device).expand(
        batch_size, max_len
    ) < lengths.unsqueeze(1)
    return mask.float()

def plot_attention(attention_weights, text_length, mel_length, step):
    """
    绘制注意力图并返回matplotlib figure
    
    Args:
        attention_weights: 注意力权重 [mel_len, text_len]
        text_length: 实际文本长度
        mel_length: 实际mel长度  
        step: 当前训练步数
    
    Returns:
        matplotlib.figure.Figure: 注意力图
    """
    # 截取有效部分
    attention = attention_weights[:mel_length, :text_length].cpu().numpy()
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(attention, aspect='auto', origin='lower', interpolation='nearest')
    
    # 设置标签和标题
    ax.set_xlabel('文本位置 (音素)')
    ax.set_ylabel('语音位置 (mel帧)')
    ax.set_title(f'注意力对齐图 - Step {step}')
    
    # 添加颜色条
    plt.colorbar(im, ax=ax)
    
    # 调整布局
    plt.tight_layout()
    
    return fig

def train_step(model, batch, optimizer, mel_loss_fn, gate_loss_fn, device):
    """单步训练"""
    model.train()
    
    # 将数据移到设备
    phoneme_ids = batch['phoneme_ids'].to(device)
    mel_spectrograms = batch['mel_spectrograms'].to(device)
    text_lengths = batch['text_lengths'].to(device)
    mel_lengths = batch['mel_lengths'].to(device)
    stop_tokens = batch['stop_tokens'].to(device)
    
    # 前向传播
    outputs = model(phoneme_ids, mel_spectrograms)
    
    pred_mels = outputs['mel_outputs']
    pred_stop_tokens = outputs['stop_tokens']
    attention_weights = outputs['attention_weights']
    
    # 创建掩码
    max_mel_len = mel_spectrograms.size(1)
    mel_mask = create_mask_from_lengths(mel_lengths, max_mel_len)
    
    # 计算损失
    mel_loss = mel_loss_fn(pred_mels, mel_spectrograms, mel_mask.unsqueeze(-1))
    gate_loss = gate_loss_fn(torch.sigmoid(pred_stop_tokens), stop_tokens.float(), mel_mask)
    
    total_loss = mel_loss + gate_loss
    
    # 反向传播
    optimizer.zero_grad()
    total_loss.backward()
    
    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    optimizer.step()
    
    return {
        'total_loss': total_loss.item(),
        'mel_loss': mel_loss.item(),
        'gate_loss': gate_loss.item(),
        'attention_weights': attention_weights.detach()
    }

def validate_step(model, eval_dataloader, mel_loss_fn, gate_loss_fn, device):
    """验证步骤"""
    model.eval()
    total_val_loss = 0
    total_mel_loss = 0
    total_gate_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in eval_dataloader:
            phoneme_ids = batch['phoneme_ids'].to(device)
            mel_spectrograms = batch['mel_spectrograms'].to(device)
            text_lengths = batch['text_lengths'].to(device)
            mel_lengths = batch['mel_lengths'].to(device)
            stop_tokens = batch['stop_tokens'].to(device)
            
            outputs = model(phoneme_ids, mel_spectrograms)
            pred_mels = outputs['mel_outputs']
            pred_stop_tokens = outputs['stop_tokens']
            
            max_mel_len = mel_spectrograms.size(1)
            mel_mask = create_mask_from_lengths(mel_lengths, max_mel_len)
            
            mel_loss = mel_loss_fn(pred_mels, mel_spectrograms, mel_mask.unsqueeze(-1))
            gate_loss = gate_loss_fn(torch.sigmoid(pred_stop_tokens), stop_tokens.float(), mel_mask)
            
            total_val_loss += (mel_loss + gate_loss).item()
            total_mel_loss += mel_loss.item()
            total_gate_loss += gate_loss.item()
            num_batches += 1
    
    return {
        'val_loss': total_val_loss / num_batches,
        'val_mel_loss': total_mel_loss / num_batches,
        'val_gate_loss': total_gate_loss / num_batches
    }

def train_distributed(rank, world_size):
    """分布式训练主函数"""
    # 设置分布式环境
    setup_distributed(rank, world_size)
    
    # 设置设备
    device = torch.device(f'cuda:{rank}')
    
    # 将模型移到对应GPU并包装为DDP
    model.to(device)
    ddp_model = DDP(model, device_ids=[rank])
    
    # 创建分布式采样器
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    eval_sampler = DistributedSampler(eval_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    # 重新创建DataLoader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=shore_dataset.collate_fn,
        pin_memory=True,
        drop_last=True
    )
    
    eval_loader = DataLoader(
        dataset=eval_dataset,
        batch_size=eval_config['batch_size'],
        sampler=eval_sampler,
        num_workers=num_workers,
        collate_fn=shore_dataset.collate_fn,
        pin_memory=True,
        drop_last=False
    )
    
    # 只在主进程创建TensorBoard和日志
    writer = None
    if rank == 0:
        writer = SummaryWriter('logs/tensorboard')
        logging.info(f"开始分布式训练 - 世界大小: {world_size}")
    
    global_step = 0
    
    for epoch in range(epochs):
        # 设置采样器的epoch
        train_sampler.set_epoch(epoch)
        
        epoch_losses = []
        
        for step, batch in enumerate(train_loader):
            # 训练步骤
            losses = train_step(ddp_model, batch, optimizer, mel_loss, gate_loss, device)
            epoch_losses.append(losses)
            
            global_step += 1
            
            # 日志记录 (只在主进程)
            if rank == 0 and global_step % log_every_steps == 0:
                logging.info(
                    f"Epoch {epoch+1}/{epochs}, Step {step+1}/{len(train_loader)}, "
                    f"Loss: {losses['total_loss']:.4f}, "
                    f"Mel: {losses['mel_loss']:.4f}, "
                    f"Gate: {losses['gate_loss']:.4f}"
                )
                if writer:
                    writer.add_scalar('Train/TotalLoss', losses['total_loss'], global_step)
                    writer.add_scalar('Train/MelLoss', losses['mel_loss'], global_step)
                    writer.add_scalar('Train/GateLoss', losses['gate_loss'], global_step)
                    
                    # 记录注意力图 (每1000步记录一次，避免过于频繁)
                    if global_step % (log_every_steps * 10) == 0:
                        # 取第一个样本的注意力权重进行可视化
                        attention = losses['attention_weights'][0]  # [mel_len, text_len]
                        text_len = batch['text_lengths'][0].item()
                        mel_len = batch['mel_lengths'][0].item()
                        
                        # 绘制注意力图
                        fig = plot_attention(attention, text_len, mel_len, global_step)
                        writer.add_figure('Train/Attention', fig, global_step)
                        plt.close(fig)  # 释放内存
            
            # 验证 (只在主进程)
            if rank == 0 and global_step % eval_every_steps == 0:
                # 进行验证并获取一个样本的注意力权重用于可视化
                model.eval()
                with torch.no_grad():
                    # 取验证集的第一个batch进行可视化
                    val_batch = next(iter(eval_loader))
                    val_phoneme_ids = val_batch['phoneme_ids'].to(device)
                    val_mel_spectrograms = val_batch['mel_spectrograms'].to(device)
                    val_outputs = model(val_phoneme_ids, val_mel_spectrograms)
                    val_attention = val_outputs['attention_weights'][0]  # [mel_len, text_len]
                    val_text_len = val_batch['text_lengths'][0].item()
                    val_mel_len = val_batch['mel_lengths'][0].item()
                
                val_losses = validate_step(model, eval_loader, mel_loss, gate_loss, device)
                logging.info(
                    f"Validation - Loss: {val_losses['val_loss']:.4f}, "
                    f"Mel: {val_losses['val_mel_loss']:.4f}, "
                    f"Gate: {val_losses['val_gate_loss']:.4f}"
                )
                if writer:
                    writer.add_scalar('Val/TotalLoss', val_losses['val_loss'], global_step)
                    writer.add_scalar('Val/MelLoss', val_losses['val_mel_loss'], global_step)
                    writer.add_scalar('Val/GateLoss', val_losses['val_gate_loss'], global_step)
                    
                    # 记录验证集的注意力图
                    val_fig = plot_attention(val_attention, val_text_len, val_mel_len, global_step)
                    writer.add_figure('Val/Attention', val_fig, global_step)
                    plt.close(val_fig)
            
            # 保存检查点 (只在主进程)
            if rank == 0 and global_step % save_every_steps == 0:
                checkpoint = {
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'config': config
                }
                torch.save(checkpoint, f'checkpoints/checkpoint_step_{global_step}.pt')
                logging.info(f"保存检查点: checkpoint_step_{global_step}.pt")
        
        # 学习率调度
        scheduler.step()
        
        # 每个epoch结束后保存 (只在主进程)
        if rank == 0 and (epoch + 1) % save_every_epoch == 0:
            checkpoint = {
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': config
            }
            torch.save(checkpoint, f'checkpoints/checkpoint_epoch_{epoch+1}.pt')
            logging.info(f"保存epoch检查点: checkpoint_epoch_{epoch+1}.pt")
        
        # 同步所有进程
        dist.barrier()
    
    if rank == 0 and writer:
        writer.close()
        logging.info("训练完成")
    
    cleanup_distributed()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=8, help='分布式训练的GPU数量')
    args = parser.parse_args()
    
    world_size = args.world_size
    
    if world_size > 1:
        mp.spawn(train_distributed, args=(world_size,), nprocs=world_size, join=True)
    else:
        # 单GPU情况下也使用分布式框架
        train_distributed(0, 1)