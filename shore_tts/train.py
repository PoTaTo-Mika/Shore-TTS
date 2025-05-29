import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm
import time
import argparse
import warnings

# 过滤CUDNN相关警告
warnings.filterwarnings('ignore', category=UserWarning, message='.*cudnn.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*CUDNN.*')

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

def setup_distributed():
    """设置分布式训练环境，用于torchrun启动"""
    # 从环境变量获取分布式训练参数
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    # 初始化进程组
    if world_size > 1:
        dist.init_process_group(backend='nccl')
        # 设置当前进程使用的GPU
        torch.cuda.set_device(local_rank)
    
    return rank, local_rank, world_size

def cleanup_distributed():
    """清理分布式训练环境"""
    if dist.is_initialized():
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

def create_dataloader(config, rank, world_size, is_validation=False):
    """创建数据加载器"""
    data_config = config['training']['data']
    training_config = config['training']
    
    # 创建完整数据集
    full_dataset = ShoreDataset(
        mel_list_path=data_config['mel_list_path'],
        pinyin_list_path=data_config['pinyin_list_path'],
        device='cpu',  # 修改为CPU，避免多进程CUDA问题
        max_mel_length=config['model']['max_mel_len'],
        min_mel_length=100,
        enable_filter=data_config['enable_filter']
    )
    
    # 计算划分索引
    val_split = training_config.get('val_split', 0.1)
    total_size = len(full_dataset)
    val_size = 500
    train_size = total_size - val_size
    
    # 设置随机种子确保划分一致性
    torch.manual_seed(config['training']['seed'])
    
    # 划分数据集
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # 选择要使用的数据集
    dataset = val_dataset if is_validation else train_dataset
    
    # 创建分布式采样器
    if world_size > 1 and not is_validation:
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
        shuffle=(sampler is None and not is_validation),
        collate_fn=full_dataset.collate_fn,
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

    elif training_config['optimizer'] == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=1e-6
        )
    
    elif training_config['optimizer'] == 'AdaFactor':
        # AdaFactor: 内存高效的Adam替代方案
        try:
            from transformers import Adafactor
            optimizer = Adafactor(
                model.parameters(),
                lr=training_config['learning_rate'],
                scale_parameter=False,
                relative_step=False,
                warmup_init=False,
                weight_decay=1e-6
            )
        except ImportError:
            print("AdaFactor需要安装transformers库: pip install transformers")
            # 回退到RMSprop
            optimizer = optim.RMSprop(
                model.parameters(),
                lr=training_config['learning_rate'],
                weight_decay=1e-6,
                alpha=0.9
            )
    
    elif training_config['optimizer'] == 'Lion':
        # Lion: Google开发的内存高效优化器
        try:
            from lion_pytorch import Lion
            optimizer = Lion(
                model.parameters(),
                lr=training_config['learning_rate'] * 0.1,  # Lion通常需要更小的学习率
                weight_decay=1e-2  # Lion通常需要更大的weight_decay
            )
        except ImportError:
            print("Lion需要安装lion-pytorch库: pip install lion-pytorch")
            # 回退到RMSprop
            optimizer = optim.RMSprop(
                model.parameters(),
                lr=training_config['learning_rate'],
                weight_decay=1e-6,
                alpha=0.9
            )
    
    elif training_config['optimizer'] == 'RMSprop':
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=1e-6,
            alpha=0.9,
            momentum=0.9
        )
    
    elif training_config['optimizer'] == 'AdamW_8bit':
        # 8bit AdamW，大幅减少内存占用
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                model.parameters(),
                lr=training_config['learning_rate'],
                weight_decay=1e-6
            )
        except ImportError:
            print("AdamW_8bit需要安装bitsandbytes库: pip install bitsandbytes")
            # 回退到RMSprop
            optimizer = optim.RMSprop(
                model.parameters(),
                lr=training_config['learning_rate'],
                weight_decay=1e-6,
                alpha=0.9
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
        
        # 将数据移动到GPU
        device = next(model.parameters()).device
        padded_mels = padded_mels.to(device)
        padded_phoneme_ids = padded_phoneme_ids.to(device)
        mel_lengths = mel_lengths.to(device)
        phoneme_lengths = phoneme_lengths.to(device)
        
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

def validate_epoch(model, dataloader, config, rank):
    """验证一个epoch"""
    model.eval()
    total_loss = 0
    total_mel_loss = 0
    total_postnet_loss = 0
    total_stop_loss = 0
    
    # 只在主进程显示进度条
    if rank == 0:
        pbar = tqdm(dataloader, desc='Validation')
    else:
        pbar = dataloader
    
    with torch.no_grad():
        for step, batch_data in enumerate(pbar):
            padded_mels, padded_phoneme_ids, mel_lengths, phoneme_lengths = batch_data
            
            # 将数据移动到GPU
            device = next(model.parameters()).device
            padded_mels = padded_mels.to(device)
            padded_phoneme_ids = padded_phoneme_ids.to(device)
            mel_lengths = mel_lengths.to(device)
            phoneme_lengths = phoneme_lengths.to(device)
            
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
            
            # 统计损失
            total_loss += loss.item()
            total_mel_loss += mel_loss.item()
            total_postnet_loss += postnet_loss.item()
            total_stop_loss += stop_loss.item()
            
            # 更新进度条
            if rank == 0:
                pbar.set_postfix({
                    'Val Loss': f'{loss.item():.4f}',
                    'Mel': f'{mel_loss.item():.4f}',
                    'Post': f'{postnet_loss.item():.4f}',
                    'Stop': f'{stop_loss.item():.4f}'
                })
    
    # 计算平均损失
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_mel_loss = total_mel_loss / num_batches
    avg_postnet_loss = total_postnet_loss / num_batches
    avg_stop_loss = total_stop_loss / num_batches
    
    return avg_loss, avg_mel_loss, avg_postnet_loss, avg_stop_loss

def train_process(config, args):
    """训练进程"""
    try:
        # 设置分布式环境
        rank, local_rank, world_size = setup_distributed()
        
        # 设置日志
        setup_logging(rank)
        
        # 设置随机种子
        set_random_seed(config['training']['seed'])
        
        # 设置设备
        device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() and world_size > 1 else 'cuda' if torch.cuda.is_available() else 'cpu')
        
        if rank == 0:
            logging.info(f"开始训练，使用设备: {device}")
            logging.info(f"分布式训练: 世界大小={world_size}, 当前进程rank={rank}, 本地rank={local_rank}")
            logging.info(f"配置: {json.dumps(config, indent=2, ensure_ascii=False)}")
        
        # 创建数据加载器
        train_dataloader, train_sampler = create_dataloader(config, rank, world_size, is_validation=False)
        val_dataloader, _ = create_dataloader(config, rank, world_size, is_validation=True)
        
        if rank == 0:
            logging.info(f"训练集大小: {len(train_dataloader.dataset)}")
            logging.info(f"验证集大小: {len(val_dataloader.dataset)}")
            logging.info(f"训练批次数量: {len(train_dataloader)}")
            logging.info(f"验证批次数量: {len(val_dataloader)}")
        
        # 创建模型
        model = create_model(config)
        model = model.to(device)
        
        # 计算模型参数
        if rank == 0:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logging.info(f"模型参数数量: {total_params:,} (可训练: {trainable_params:,})")
        
        # 创建分布式模型
        if world_size > 1:
            model = DDP(model, device_ids=[local_rank])
        
        # 创建优化器
        optimizer = create_optimizer(model, config)
        
        # 创建学习率调度器
        total_steps = len(train_dataloader) * config['training']['epochs']
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
        
        # 初始化早停变量
        best_val_loss = float('inf')
        patience_counter = 0
        early_stopping_patience = config['training'].get('early_stopping_patience', 10)
        val_interval = config['training'].get('val_interval', 5)
        save_best_model = config['training'].get('save_best_model', True)
        
        # 训练循环
        for epoch in range(start_epoch, config['training']['epochs']):
            # 设置分布式采样器的epoch（用于打乱数据）
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            
            # 训练一个epoch
            start_time = time.time()
            avg_loss, avg_mel_loss, avg_postnet_loss, avg_stop_loss = train_epoch(
                model, train_dataloader, optimizer, scheduler, None, epoch, config, rank
            )
            epoch_time = time.time() - start_time
            
            # 日志输出
            if rank == 0:
                logging.info(
                    f"Epoch {epoch}/{config['training']['epochs']} - "
                    f"Train Loss: {avg_loss:.4f}, "
                    f"Mel: {avg_mel_loss:.4f}, "
                    f"Post: {avg_postnet_loss:.4f}, "
                    f"Stop: {avg_stop_loss:.4f}, "
                    f"Time: {epoch_time:.2f}s"
                )
            
            # 验证
            if (epoch + 1) % val_interval == 0:
                val_start_time = time.time()
                val_loss, val_mel_loss, val_postnet_loss, val_stop_loss = validate_epoch(
                    model, val_dataloader, config, rank
                )
                val_time = time.time() - val_start_time
                
                if rank == 0:
                    logging.info(
                        f"Validation - "
                        f"Loss: {val_loss:.4f}, "
                        f"Mel: {val_mel_loss:.4f}, "
                        f"Post: {val_postnet_loss:.4f}, "
                        f"Stop: {val_stop_loss:.4f}, "
                        f"Time: {val_time:.2f}s"
                    )
                
                # 保存最佳模型
                if save_best_model and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    if rank == 0:
                        best_model_path = os.path.join(save_dir, "best_model.pt")
                        save_checkpoint(
                            model, optimizer, scheduler, epoch,
                            epoch * len(train_dataloader), val_loss, best_model_path, rank
                        )
                        logging.info(f"保存最佳模型，验证损失: {val_loss:.4f}")
                else:
                    patience_counter += 1
                
                # 早停检查
                if patience_counter >= early_stopping_patience:
                    if rank == 0:
                        logging.info(f"早停触发，验证损失连续{early_stopping_patience}次验证没有改善")
                    break
            
            # 保存定期检查点
            if rank == 0 and (epoch + 1) % 1 == 0:
                checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt")
                save_checkpoint(
                    model, optimizer, scheduler, epoch, 
                    epoch * len(train_dataloader), avg_loss, checkpoint_path, rank
                )
        
        # 保存最终模型
        if rank == 0:
            final_checkpoint_path = os.path.join(save_dir, "final_model.pt")
            save_checkpoint(
                model, optimizer, scheduler, config['training']['epochs'] - 1,
                config['training']['epochs'] * len(train_dataloader), avg_loss, 
                final_checkpoint_path, rank
            )
            logging.info("训练完成！")
    
    except Exception as e:
        logging.error(f"训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理分布式环境
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
    
    # 直接运行训练进程（torchrun会负责进程管理）
    train_process(config, args)

if __name__ == "__main__":
    main()

