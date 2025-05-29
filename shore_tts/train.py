import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm
import time
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)  # 获取项目根目录
sys.path.insert(0, root_dir)

from shore_tts.modules.model import ShoreTTS
from shore_tts.datasets.shore_datasets import ShoreDataset

import json
import logging

def setup_logging(rank=0):
    """设置日志系统"""
    if rank == 0:  # 只在主进程输出日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        logging.basicConfig(level=logging.WARNING)

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config

def set_random_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)

def setup_distributed(rank, world_size, dist_url, dist_backend):
    """设置分布式训练环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = dist_url.split(':')[-1]
    
    # 初始化进程组
    dist.init_process_group(
        backend=dist_backend,
        rank=rank,
        world_size=world_size
    )
    
    # 设置当前进程使用的GPU
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """清理分布式训练环境"""
    dist.destroy_process_group()

def create_model(config):
    """创建模型"""
    model_config = config['model']
    model = ShoreTTS(
        vocab_size=model_config['vocab_size'],
        text_embedding_dim=model_config['text_embedding_dim'],
        text_hidden_dim=model_config['text_hidden_dim'],
        text_n_layers=model_config['text_n_layers'],
        text_n_heads=model_config['text_n_heads'],
        text_d_model=model_config['text_d_model'],
        n_mels=model_config['n_mels'],
        decoder_d_model=model_config['decoder_d_model'],
        decoder_n_heads=model_config['decoder_n_heads'],
        decoder_n_layers=model_config['decoder_n_layers'],
        decoder_d_ff=model_config['decoder_d_ff'],
        postnet_n_layers=model_config['postnet_n_layers'],
        postnet_kernel_size=model_config['postnet_kernel_size'],
        postnet_n_channels=model_config['postnet_n_channels'],
        dropout=model_config['dropout'],
        max_text_len=model_config['max_text_len'],
        max_mel_len=model_config['max_mel_len']
    )
    return model

def create_dataloader(config, rank, world_size):
    """创建数据加载器"""
    data_config = config['training']['data']
    training_config = config['training']
    
    # 创建数据集
    dataset = ShoreDataset(
        mel_list_path=data_config['mel_list_path'],
        pinyin_list_path=data_config['pinyin_list_path'],
        device=f'cuda:{rank}',
        max_mel_length=config['model']['max_mel_len'],
        min_mel_length=100
    )
    
    # 创建分布式采样器
    if config['training']['distributed']['enabled'] and world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
    else:
        sampler = None
    
    # 创建数据加载器
    dataloader = data.DataLoader(
        dataset,
        batch_size=training_config['batch_size'],
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=dataset.collate_fn,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader, sampler

def create_optimizer(model, config):
    """创建优化器"""
    training_config = config['training']
    
    if training_config['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=1e-6
        )
    elif training_config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=training_config['learning_rate']
        )
    else:
        raise ValueError(f"不支持的优化器: {training_config['optimizer']}")
    
    return optimizer

def create_scheduler(optimizer, config, total_steps):
    """创建学习率调度器"""
    # 使用余弦退火调度器
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=1e-6
    )
    return scheduler

def compute_loss(mel_outputs, mel_outputs_postnet, stop_outputs, mel_targets, mel_lengths):
    """计算损失"""
    batch_size = mel_targets.shape[0]
    device = mel_targets.device
    
    mel_criterion = nn.MSELoss(reduction='none')
    stop_criterion = nn.BCELoss(reduction='none')
    
    total_mel_loss = 0
    total_postnet_loss = 0
    total_stop_loss = 0
    
    for i in range(batch_size):
        actual_length = mel_lengths[i].item()
        
        # Mel损失（只计算有效长度）
        mel_loss = mel_criterion(
            mel_outputs[i, :actual_length, :],
            mel_targets[i, :actual_length, :]
        ).mean()
        
        postnet_loss = mel_criterion(
            mel_outputs_postnet[i, :actual_length, :],
            mel_targets[i, :actual_length, :]
        ).mean()
        
        # 停止标记损失
        stop_target = torch.zeros(actual_length, 1, device=device)
        stop_target[-1, 0] = 1.0  # 最后一帧为停止标记
        
        stop_loss = stop_criterion(
            stop_outputs[i, :actual_length, :],
            stop_target
        ).mean()
        
        total_mel_loss += mel_loss
        total_postnet_loss += postnet_loss
        total_stop_loss += stop_loss
    
    # 平均损失
    total_mel_loss = total_mel_loss / batch_size
    total_postnet_loss = total_postnet_loss / batch_size
    total_stop_loss = total_stop_loss / batch_size
    
    # 总损失
    total_loss = total_mel_loss * 1.1 + total_postnet_loss * 0.7 + total_stop_loss * 1.2
    
    return total_loss, total_mel_loss, total_postnet_loss, total_stop_loss

def save_checkpoint(model, optimizer, scheduler, epoch, step, loss, save_path, rank):
    """保存检查点"""
    if rank == 0:  # 只在主进程保存
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 如果是DDP模型，需要保存module的状态
        model_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'loss': loss
        }
        
        torch.save(checkpoint, save_path)
        logging.info(f"检查点已保存到: {save_path}")

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """加载检查点"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 加载模型状态
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # 加载优化器状态
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 加载调度器状态
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['epoch'], checkpoint['step'], checkpoint['loss']

def train_epoch(model, dataloader, optimizer, scheduler, criterion, epoch, config, rank):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_mel_loss = 0
    total_postnet_loss = 0
    total_stop_loss = 0
    
    gradient_accumulation_steps = config['training']['gradient_accumulation_steps']
    
    # 只在主进程显示进度条
    if rank == 0:
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    else:
        pbar = dataloader
    
    optimizer.zero_grad()
    
    for step, batch_data in enumerate(pbar):
        padded_mels, padded_phoneme_ids, mel_lengths, phoneme_lengths = batch_data
        
        # 转换mel格式: [batch, n_mels, max_len] -> [batch, max_len, n_mels]
        mel_targets = padded_mels.transpose(1, 2)
        
        # 前向传播
        mel_outputs, mel_outputs_postnet, stop_outputs, attention_weights = model(
            phoneme_ids=padded_phoneme_ids,
            phoneme_lengths=phoneme_lengths,
            mel_target=mel_targets
        )
        
        # 计算损失
        loss, mel_loss, postnet_loss, stop_loss = compute_loss(
            mel_outputs, mel_outputs_postnet, stop_outputs, mel_targets, mel_lengths
        )
        
        # 归一化损失（用于梯度累积）
        loss = loss / gradient_accumulation_steps
        
        # 反向传播
        loss.backward()
        
        # 累积梯度
        if (step + 1) % gradient_accumulation_steps == 0:
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 更新参数
            optimizer.step()
            if scheduler:
                scheduler.step()
            
            # 清零梯度
            optimizer.zero_grad()
        
        # 统计损失
        total_loss += loss.item() * gradient_accumulation_steps
        total_mel_loss += mel_loss.item()
        total_postnet_loss += postnet_loss.item()
        total_stop_loss += stop_loss.item()
        
        # 更新进度条
        if rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'Loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
                'Mel': f'{mel_loss.item():.4f}',
                'Post': f'{postnet_loss.item():.4f}',
                'Stop': f'{stop_loss.item():.4f}',
                'LR': f'{current_lr:.2e}'
            })
    
    # 计算平均损失
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_mel_loss = total_mel_loss / num_batches
    avg_postnet_loss = total_postnet_loss / num_batches
    avg_stop_loss = total_stop_loss / num_batches
    
    return avg_loss, avg_mel_loss, avg_postnet_loss, avg_stop_loss

def train_process(rank, world_size, config, args):
    """分布式训练进程"""
    try:
        # 设置日志
        setup_logging(rank)
        
        # 设置随机种子
        set_random_seed(config['training']['seed'])
        
        # 设置分布式环境
        if config['training']['distributed']['enabled'] and world_size > 1:
            setup_distributed(
                rank, 
                world_size, 
                config['training']['distributed']['dist_url'],
                config['training']['distributed']['dist_backend']
            )
        
        # 设置设备
        device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
        
        if rank == 0:
            logging.info(f"开始训练，使用设备: {device}")
            logging.info(f"配置: {json.dumps(config, indent=2, ensure_ascii=False)}")
        
        # 创建数据加载器
        dataloader, sampler = create_dataloader(config, rank, world_size)
        
        if rank == 0:
            logging.info(f"数据集大小: {len(dataloader.dataset)}")
            logging.info(f"批次数量: {len(dataloader)}")
        
        # 创建模型
        model = create_model(config)
        model = model.to(device)
        
        # 计算模型参数
        if rank == 0:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logging.info(f"模型参数数量: {total_params:,} (可训练: {trainable_params:,})")
        
        # 创建分布式模型
        if config['training']['distributed']['enabled'] and world_size > 1:
            model = DDP(model, device_ids=[rank])
        
        # 创建优化器
        optimizer = create_optimizer(model, config)
        
        # 创建学习率调度器
        total_steps = len(dataloader) * config['training']['epochs']
        scheduler = create_scheduler(optimizer, config, total_steps)
        
        # 加载检查点（如果存在）
        start_epoch = 0
        if args.resume and os.path.exists(args.resume):
            if rank == 0:
                logging.info(f"从检查点恢复训练: {args.resume}")
            start_epoch, _, _ = load_checkpoint(args.resume, model, optimizer, scheduler)
            start_epoch += 1
        
        # 创建保存目录
        save_dir = config['training']['save_dir']
        if rank == 0:
            os.makedirs(save_dir, exist_ok=True)
        
        # 训练循环
        for epoch in range(start_epoch, config['training']['epochs']):
            # 设置分布式采样器的epoch（用于打乱数据）
            if sampler is not None:
                sampler.set_epoch(epoch)
            
            # 训练一个epoch
            start_time = time.time()
            avg_loss, avg_mel_loss, avg_postnet_loss, avg_stop_loss = train_epoch(
                model, dataloader, optimizer, scheduler, None, epoch, config, rank
            )
            epoch_time = time.time() - start_time
            
            # 日志输出
            if rank == 0:
                logging.info(
                    f"Epoch {epoch}/{config['training']['epochs']} - "
                    f"Loss: {avg_loss:.4f}, "
                    f"Mel: {avg_mel_loss:.4f}, "
                    f"Post: {avg_postnet_loss:.4f}, "
                    f"Stop: {avg_stop_loss:.4f}, "
                    f"Time: {epoch_time:.2f}s"
                )
            
            # 保存检查点
            if rank == 0 and (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt")
                save_checkpoint(
                    model, optimizer, scheduler, epoch, 
                    epoch * len(dataloader), avg_loss, checkpoint_path, rank
                )
        
        # 保存最终模型
        if rank == 0:
            final_checkpoint_path = os.path.join(save_dir, "final_model.pt")
            save_checkpoint(
                model, optimizer, scheduler, config['training']['epochs'] - 1,
                config['training']['epochs'] * len(dataloader), avg_loss, 
                final_checkpoint_path, rank
            )
            logging.info("训练完成！")
    
    except Exception as e:
        logging.error(f"训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理分布式环境
        if config['training']['distributed']['enabled'] and world_size > 1:
            cleanup_distributed()

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Shore TTS 训练脚本')
    parser.add_argument('--config', type=str, default='shore_tts/configs/base.json',
                        help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 分布式训练设置
    distributed_config = config['training']['distributed']
    world_size = distributed_config['world_size'] if distributed_config['enabled'] else 1
    
    if distributed_config['enabled'] and world_size > 1:
        # 多GPU分布式训练
        mp.spawn(
            train_process,
            args=(world_size, config, args),
            nprocs=world_size,
            join=True
        )
    else:
        # 单GPU或CPU训练
        train_process(0, 1, config, args)

if __name__ == "__main__":
    main()

