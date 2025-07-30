import torch
import numpy as np
import os
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class ShoreTTSDataset(Dataset):
    """
    Shore TTS 数据集类 - 专门处理音素体系的TTS数据
    
    数据格式要求：
    - 文本文件：包含音素ID序列的数字列表 (已转换为词典中的ID)
    - 音频文件：mel频谱的.npy文件
    - 文件命名：相同文件名，不同扩展名 (如 001.txt, 001.npy)
    """
    
    def __init__(self, 
                 data_root,           # 数据根目录
                 text_subdir="texts", # 文本子目录名
                 mel_subdir="mels",   # mel频谱子目录名
                 pad_token_id=0,      # 填充token的ID (通常是_pad符号的ID)
                 eos_token_id=1,      # 结束token的ID (通常是_eos符号的ID)
                 add_eos=True,        # 是否在文本序列末尾添加EOS token
                 ):
        """
        Args:
            data_root (str): 数据根目录路径
            text_subdir (str): 文本文件子目录名，默认"texts"
            mel_subdir (str): mel频谱文件子目录名，默认"mels"  
            pad_token_id (int): 填充符的token ID，默认0 (对应"_"符号)
            eos_token_id (int): 结束符的token ID，默认1 (对应"~"符号)
            add_eos (bool): 是否在文本序列末尾添加EOS token，默认True
        """
        super(ShoreTTSDataset, self).__init__()
        
        self.data_root = data_root
        self.text_dir = os.path.join(data_root, text_subdir)
        self.mel_dir = os.path.join(data_root, mel_subdir)
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.add_eos = add_eos
        
        # 验证目录存在
        if not os.path.exists(self.text_dir):
            raise FileNotFoundError(f"文本目录不存在: {self.text_dir}")
        if not os.path.exists(self.mel_dir):
            raise FileNotFoundError(f"Mel频谱目录不存在: {self.mel_dir}")
        
        # 获取所有数据文件对
        self.data_pairs = self._collect_data_pairs()
        
        print(f"数据集初始化完成:")
        print(f"  - 数据根目录: {data_root}")
        print(f"  - 文本目录: {self.text_dir}")
        print(f"  - Mel目录: {self.mel_dir}")
        print(f"  - 数据对数量: {len(self.data_pairs)}")
        print(f"  - 填充Token ID: {pad_token_id}")
        print(f"  - 结束Token ID: {eos_token_id}")
        print(f"  - 添加EOS: {add_eos}")
    
    def _collect_data_pairs(self):
        """
        收集所有有效的文本-音频数据对
        
        Returns:
            list: 包含(text_file_path, mel_file_path)元组的列表
        """
        data_pairs = []
        
        # 获取所有文本文件
        text_files = [f for f in os.listdir(self.text_dir) if f.endswith('.txt')]
        
        for text_file in text_files:
            # 提取文件名(不含扩展名)
            basename = os.path.splitext(text_file)[0]
            
            # 构建对应的mel文件路径
            mel_file = basename + '.npy'
            text_path = os.path.join(self.text_dir, text_file)
            mel_path = os.path.join(self.mel_dir, mel_file)
            
            # 检查对应的mel文件是否存在
            if os.path.exists(mel_path):
                data_pairs.append((text_path, mel_path))
            else:
                print(f"警告: 找不到对应的mel文件: {mel_path}")
        
        if len(data_pairs) == 0:
            raise ValueError("未找到任何有效的文本-音频数据对")
        
        # 按文件名排序，确保数据加载的一致性
        data_pairs.sort(key=lambda x: os.path.basename(x[0]))
        
        return data_pairs
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        """
        获取单个数据样本
        
        Args:
            idx (int): 数据索引
            
        Returns:
            dict: 包含以下键值对的字典
                - 'phoneme_ids': 音素ID序列张量 [seq_len]
                - 'mel_spectrogram': mel频谱张量 [mel_len, n_mels]
                - 'text_length': 文本序列长度 (标量)
                - 'mel_length': mel频谱长度 (标量)
        """
        text_path, mel_path = self.data_pairs[idx]
        
        try:
            # 加载文本数据 (音素ID序列)
            phoneme_ids = self._load_text_data(text_path)
            
            # 加载mel频谱数据
            mel_spectrogram = self._load_mel_data(mel_path)
            
            return {
                'phoneme_ids': phoneme_ids,
                'mel_spectrogram': mel_spectrogram,
                'text_length': len(phoneme_ids),
                'mel_length': mel_spectrogram.shape[0]  # 时间维度长度
            }
            
        except Exception as e:
            print(f"加载数据时出错 - 文本: {text_path}, Mel: {mel_path}")
            print(f"错误信息: {str(e)}")
            raise e
    
    def _load_text_data(self, text_path):
        """
        加载文本数据 (音素ID序列)
        
        Args:
            text_path (str): 文本文件路径
            
        Returns:
            torch.Tensor: 音素ID序列张量 [seq_len]
        """
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # 解析音素ID序列，优先检查Python列表格式
            if content.startswith('[') and content.endswith(']'):
                # Python列表格式: [1,2,3,4,5]
                phoneme_ids = eval(content)
            elif ',' in content and not (content.startswith('[') and content.endswith(']')):
                # 逗号分隔格式: "1,2,3,4,5"
                phoneme_ids = [int(x.strip()) for x in content.split(',') if x.strip()]
            elif ' ' in content:
                # 空格分隔格式: "1 2 3 4 5"
                phoneme_ids = [int(x.strip()) for x in content.split() if x.strip()]
            else:
                # 单个数字或其他格式
                phoneme_ids = [int(content)]
                
            # 转换为张量
            phoneme_ids = torch.tensor(phoneme_ids, dtype=torch.long)
            
            # 添加EOS token (如果需要)
            if self.add_eos:
                eos_tensor = torch.tensor([self.eos_token_id], dtype=torch.long)
                phoneme_ids = torch.cat([phoneme_ids, eos_tensor], dim=0)
            
            return phoneme_ids
            
        except Exception as e:
            raise ValueError(f"无法解析文本文件 {text_path}: {str(e)}")
    
    def _load_mel_data(self, mel_path):
        """
        加载mel频谱数据
        
        Args:
            mel_path (str): mel频谱文件路径 (.npy)
            
        Returns:
            torch.Tensor: mel频谱张量 [mel_len, n_mels]
        """
        try:
            # 加载numpy数组
            mel_array = np.load(mel_path)
            
            # 转换为张量
            mel_tensor = torch.tensor(mel_array, dtype=torch.float32)
            
            # 确保维度正确: [time_steps, n_mels]
            if mel_tensor.dim() == 1:
                # 如果是1D，假设是单帧，扩展为 [1, n_mels]
                mel_tensor = mel_tensor.unsqueeze(0)
            elif mel_tensor.dim() == 2:
                # 检查维度顺序，确保是 [time_steps, n_mels]
                if mel_tensor.shape[0] < mel_tensor.shape[1]:
                    # 如果第一个维度较小，可能需要转置
                    # 但这里假设数据已经是正确格式 [time_steps, n_mels]
                    pass
            else:
                raise ValueError(f"Mel频谱维度不正确: {mel_tensor.shape}")
            
            return mel_tensor
            
        except Exception as e:
            raise ValueError(f"无法加载mel频谱文件 {mel_path}: {str(e)}")
    
    def get_sample_info(self, idx):
        """
        获取样本信息 (用于调试)
        
        Args:
            idx (int): 样本索引
            
        Returns:
            dict: 样本信息
        """
        text_path, mel_path = self.data_pairs[idx]
        sample = self[idx]
        
        return {
            'index': idx,
            'text_file': os.path.basename(text_path),
            'mel_file': os.path.basename(mel_path),
            'text_length': sample['text_length'],
            'mel_length': sample['mel_length'],
            'phoneme_ids_sample': sample['phoneme_ids'][:10].tolist(),  # 前10个音素ID
            'mel_shape': sample['mel_spectrogram'].shape
        }

def collate_fn(batch):
    """
    Args:
        batch (list): 包含多个样本字典的列表
        
    Returns:
        dict: 批处理后的数据字典
            - 'phoneme_ids': 填充后的音素序列 [batch_size, max_text_len]
            - 'mel_spectrograms': 填充后的mel频谱 [batch_size, max_mel_len, n_mels]
            - 'text_lengths': 各样本的实际文本长度 [batch_size]
            - 'mel_lengths': 各样本的实际mel长度 [batch_size]
            - 'stop_tokens': 停止标记 [batch_size, max_mel_len]
    """
    
    # 提取各个字段
    phoneme_ids_list = [item['phoneme_ids'] for item in batch]
    mel_spectrograms_list = [item['mel_spectrogram'] for item in batch]
    text_lengths = torch.tensor([item['text_length'] for item in batch], dtype=torch.long)
    mel_lengths = torch.tensor([item['mel_length'] for item in batch], dtype=torch.long)
    
    # 填充音素序列 (使用pad_token_id=0进行填充)
    # pad_sequence默认填充值为0，batch_first=True确保输出形状为[batch_size, max_len]
    phoneme_ids_padded = pad_sequence(
        phoneme_ids_list, 
        batch_first=True, 
        padding_value=0  # 使用pad_token_id作为填充值
    )
    
    # 填充mel频谱序列
    # 对于mel频谱，我们需要在时间维度上填充，填充值为0.0
    mel_spectrograms_padded = pad_sequence(
        mel_spectrograms_list,
        batch_first=True,
        padding_value=0.0  # mel频谱填充值为0.0
    )
    
    # 生成stop tokens
    # stop_tokens是一个与mel长度相同的序列，除了最后一帧为1，其他都为0
    stop_tokens_list = []
    for mel_len in mel_lengths:
        # 创建stop token序列：[0, 0, ..., 0, 1]
        stop_tokens = torch.zeros(mel_len, dtype=torch.long)
        if mel_len > 0:
            stop_tokens[-1] = 1  # 最后一帧设为1，表示停止
        stop_tokens_list.append(stop_tokens)
    
    # 填充stop tokens序列
    stop_tokens_padded = pad_sequence(
        stop_tokens_list,
        batch_first=True,
        padding_value=0  # 填充值为0
    )
    
    return {
        'phoneme_ids': phoneme_ids_padded,         # [batch_size, max_text_len]
        'mel_spectrograms': mel_spectrograms_padded, # [batch_size, max_mel_len, n_mels]
        'text_lengths': text_lengths,              # [batch_size]
        'mel_lengths': mel_lengths,                # [batch_size]
        'stop_tokens': stop_tokens_padded          # [batch_size, max_mel_len]
    }

# 便捷函数：创建DataLoader
def create_dataloader(data_root, 
                     batch_size=32,
                     shuffle=True,
                     num_workers=4,
                     text_subdir="texts",
                     mel_subdir="mels",
                     pad_token_id=0,
                     eos_token_id=1,
                     add_eos=True):
    """
    创建Shore TTS数据加载器的便捷函数
    
    Args:
        data_root (str): 数据根目录
        batch_size (int): 批大小，默认32
        shuffle (bool): 是否打乱数据，默认True
        num_workers (int): 数据加载工作进程数，默认4
        text_subdir (str): 文本子目录，默认"texts"
        mel_subdir (str): mel子目录，默认"mels"
        pad_token_id (int): 填充token ID，默认0
        eos_token_id (int): 结束token ID，默认1
        add_eos (bool): 是否添加EOS，默认True
        
    Returns:
        torch.utils.data.DataLoader: 配置好的数据加载器
    """
    from torch.utils.data import DataLoader
    
    # 创建数据集
    dataset = ShoreTTSDataset(
        data_root=data_root,
        text_subdir=text_subdir,
        mel_subdir=mel_subdir,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        add_eos=add_eos
    )
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,  # 使用自定义的批处理函数
        pin_memory=True,        # 加速GPU传输
        drop_last=False         # 保留最后一个不完整的批次
    )
    
    return dataloader

