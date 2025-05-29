#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清理mel.pt文件和对应.lab文件的脚本
搜索所有.pt文件，检查是否为mel文件，删除长度不在指定范围内的文件
"""

import os
import torch
import argparse
from pathlib import Path


def is_mel_file(tensor):
    """
    判断tensor是否为mel频谱文件
    通常mel频谱的特征：
    - 2D或3D tensor
    - 第一个维度通常是mel频带数（80, 128等）
    - 第二个维度是时间步长
    """
    if not isinstance(tensor, torch.Tensor):
        return False
    
    # 检查维度数
    if len(tensor.shape) < 2:
        return False
    
    # 对于2D tensor: [mel_dim, time_steps]
    if len(tensor.shape) == 2:
        mel_dim, time_steps = tensor.shape
        # 常见的mel维度范围
        if 40 <= mel_dim <= 512 and time_steps > 10:
            return True
    
    # 对于3D tensor: [batch, mel_dim, time_steps] 或 [1, mel_dim, time_steps]
    elif len(tensor.shape) == 3:
        if tensor.shape[0] == 1:  # batch size为1
            mel_dim, time_steps = tensor.shape[1], tensor.shape[2]
            if 40 <= mel_dim <= 512 and time_steps > 10:
                return True
        else:
            # 也可能是 [time_steps, mel_dim, 1] 这种格式
            pass
    
    return False


def check_mel_length(pt_path, min_length=100, max_length=3000):
    """
    检查.pt文件是否为mel文件，以及长度是否在指定范围内
    
    Args:
        pt_path: .pt文件路径
        min_length: 最小长度
        max_length: 最大长度
    
    Returns:
        tuple: (is_mel, is_valid_length, mel_length, error_msg)
    """
    try:
        # 加载.pt文件
        tensor = torch.load(pt_path, map_location='cpu')
        
        # 检查是否为mel文件
        if not is_mel_file(tensor):
            return False, False, 0, "不是mel文件"
        
        # 获取时间步长度
        if len(tensor.shape) == 2:
            mel_length = tensor.shape[1]  # [mel_dim, time_steps]
        elif len(tensor.shape) == 3 and tensor.shape[0] == 1:
            mel_length = tensor.shape[2]  # [1, mel_dim, time_steps]
        else:
            return True, False, 0, f"mel tensor形状异常: {tensor.shape}"
        
        # 检查是否包含异常值
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            return True, False, mel_length, "包含NaN或Inf值"
        
        # 检查长度是否在范围内
        if mel_length < min_length or mel_length > max_length:
            return True, False, mel_length, f"长度超出范围: {mel_length} (范围: {min_length}-{max_length})"
        
        return True, True, mel_length, "有效"
        
    except Exception as e:
        return False, False, 0, f"加载失败: {str(e)}"


def find_corresponding_lab_file(pt_path):
    """
    查找对应的.lab文件
    
    Args:
        pt_path: .pt文件路径
    
    Returns:
        str or None: 对应的.lab文件路径，如果不存在则返回None
    """
    # 将.pt替换为.lab
    lab_path = str(pt_path).replace('.pt', '.lab')
    if os.path.exists(lab_path):
        return lab_path
    
    return None


def cleanup_mel_files(directory, min_length=100, max_length=3000, dry_run=False):
    """
    清理指定目录下的mel文件
    
    Args:
        directory: 目标目录
        min_length: 最小长度
        max_length: 最大长度
        dry_run: 是否只是预览，不实际删除
    """
    directory = Path(directory)
    if not directory.exists():
        print(f"错误: 目录不存在 {directory}")
        return
    
    # 查找所有.pt文件
    print("正在搜索.pt文件...")
    pt_files = list(directory.rglob("*.pt"))
    
    if not pt_files:
        print(f"在目录 {directory} 中未找到.pt文件")
        return
    
    print(f"找到 {len(pt_files)} 个.pt文件，正在检查哪些是mel文件...")
    print(f"长度范围: {min_length} - {max_length}")
    print(f"模式: {'预览模式（不会实际删除）' if dry_run else '删除模式'}")
    print("-" * 80)
    
    deleted_count = 0
    kept_count = 0
    error_count = 0
    non_mel_count = 0
    
    for i, pt_path in enumerate(pt_files):
        if i % 100 == 0 and i > 0:
            print(f"已处理 {i}/{len(pt_files)} 个文件...")
        
        # 检查是否为mel文件以及长度
        is_mel, is_valid_length, mel_length, error_msg = check_mel_length(pt_path, min_length, max_length)
        
        if not is_mel:
            non_mel_count += 1
            continue  # 跳过非mel文件，不输出信息
        
        if is_valid_length:
            kept_count += 1
            print(f"✓ 保留: {pt_path.name} (长度: {mel_length})")
        else:
            # 查找对应的lab文件
            lab_path = find_corresponding_lab_file(pt_path)
            
            if not dry_run:
                try:
                    # 删除mel文件
                    os.remove(pt_path)
                    print(f"✗ 已删除mel: {pt_path.name} (原因: {error_msg})")
                    
                    # 删除对应的lab文件（如果存在）
                    if lab_path:
                        os.remove(lab_path)
                        print(f"✗ 已删除lab: {Path(lab_path).name}")
                    
                    deleted_count += 1
                    
                except Exception as e:
                    print(f"❌ 删除失败: {pt_path.name} - {str(e)}")
                    error_count += 1
            else:
                print(f"✗ 将删除mel: {pt_path.name} (原因: {error_msg})")
                if lab_path:
                    print(f"✗ 将删除lab: {Path(lab_path).name}")
                deleted_count += 1
    
    print("-" * 80)
    print(f"统计结果:")
    print(f"  总共.pt文件: {len(pt_files)}")
    print(f"  识别为mel文件: {kept_count + deleted_count}")
    print(f"  非mel文件: {non_mel_count}")
    print(f"  保留的mel文件: {kept_count}")
    print(f"  {'将删除' if dry_run else '已删除'}的mel文件: {deleted_count}")
    if error_count > 0:
        print(f"  删除失败的文件: {error_count}")


def main():
    parser = argparse.ArgumentParser(description="清理mel.pt文件和对应.lab文件")
    parser.add_argument("directory", help="要处理的目录路径")
    parser.add_argument("--min-length", type=int, default=100, help="最小长度 (默认: 100)")
    parser.add_argument("--max-length", type=int, default=3000, help="最大长度 (默认: 3000)")
    parser.add_argument("--dry-run", action="store_true", help="预览模式，不实际删除文件")
    
    args = parser.parse_args()
    
    print("Mel文件清理工具")
    print("=" * 80)
    
    cleanup_mel_files(
        directory=args.directory,
        min_length=args.min_length,
        max_length=args.max_length,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main() 