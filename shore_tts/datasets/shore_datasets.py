import os
import sys
import torch
import torch.utils.data as data
import numpy as np

# 设置根目录，确保能正确导入模块
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))  # 获取项目根目录
sys.path.insert(0, root_dir)

from tools.phoneme_to_id import pinyin_to_ids

class ShoreDataset(data.Dataset):
    def __init__(self, 
                 mel_list_path,
                 pinyin_list_path,
                 device = 'cuda',
                 max_mel_length = 3000, # 主要是mel容易爆显存
                 min_mel_length = 100 # 最小mel长度
    ):
        self.device = device
        self.max_mel_length = max_mel_length
        self.min_mel_length = min_mel_length
        
        with open(mel_list_path, 'r', encoding='utf-8') as f:
            # mel_list = ['mel/1.pt', 'mel/2.pt', 'mel/3.pt']
            self.mel_list = [line.strip() for line in f.readlines()]
        with open(pinyin_list_path, 'r', encoding='utf-8') as f:
            self.pinyin_list = [line.strip().split() for line in f.readlines()]
            # pinyin_list = [['ni3','hao3'], ['shi4', 'jie4']]
        
        print(f"原始mel总数量: {len(self.mel_list)}")
        print(f"原始文本总数量: {len(self.pinyin_list)}")
        
        # 检查数据一致性并过滤异常数据
        self._filter_bad_data()
        
        print(f"过滤后mel总数量: {len(self.mel_list)}")
        print(f"过滤后文本总数量: {len(self.pinyin_list)}")

    def __len__(self):
        return len(self.mel_list)
    
    def __getitem__(self, index):
        # 我们通过列表读取mel和phoneme的路径
        mel = torch.load(self.mel_list[index])
        pinyin = self.pinyin_list[index]

        # 将拼音列表转换为音素ID列表
        phoneme_ids = pinyin_to_ids(pinyin)
        # 转换为tensor
        phoneme_ids = torch.tensor(phoneme_ids, dtype=torch.long).to(self.device)
        
        # 检查数据是否有问题，如果有问题则尝试获取下一个样本
        is_valid, mel, phoneme_ids = self.process_special_case(mel, phoneme_ids)
        
        if not is_valid:
            # 如果当前样本有问题，尝试获取下一个样本
            # 为了避免无限递归，我们最多尝试10次
            for attempt in range(10):
                try:
                    next_index = (index + attempt + 1) % len(self.mel_list)
                    mel = torch.load(self.mel_list[next_index])
                    pinyin = self.pinyin_list[next_index]
                    phoneme_ids = pinyin_to_ids(pinyin)
                    phoneme_ids = torch.tensor(phoneme_ids, dtype=torch.long).to(self.device)
                    
                    is_valid, mel, phoneme_ids = self.process_special_case(mel, phoneme_ids)
                    if is_valid:
                        break
                except Exception as e:
                    print(f"尝试替代样本{next_index}时出错: {e}")
                    continue
            
            # 如果尝试了10次都没有找到有效样本，返回一个默认的小样本
            if not is_valid:
                print(f"警告: 无法找到有效的替代样本，返回默认样本")
                # 创建一个最小的有效样本
                mel = torch.zeros(128, self.min_mel_length).to(self.device)
                phoneme_ids = torch.tensor([1, 2, 3], dtype=torch.long).to(self.device)  # 简单的默认音素序列
        
        return mel, phoneme_ids
    
    def process_special_case(self, mel, phoneme):
        """
        处理特殊情况，过滤掉可能导致问题的数据
        返回: (is_valid, mel, phoneme)
        """
        # 检查mel长度是否在合理范围内
        if len(mel.shape) == 2:  # [mel_dim, time_steps]
            mel_length = mel.shape[1]
        elif len(mel.shape) == 3:  # [1, mel_dim, time_steps] 或其他
            mel_length = mel.shape[-1]
        else:
            print(f"警告: mel形状异常 {mel.shape}")
            return False, mel, phoneme
        
        # 检查mel长度范围
        if mel_length > self.max_mel_length or mel_length < self.min_mel_length:
            print(f"跳过: mel长度有问题（过长或过短）")
            return False, mel, phoneme
        
        # 检查phoneme是否为空或异常
        if phoneme is None or len(phoneme) == 0:
            print(f"跳过: phoneme为空")
            return False, mel, phoneme
        
        return True, mel, phoneme
    
    def _filter_bad_data(self):
        """
        预先过滤掉明显有问题的数据
        """
        print("开始过滤异常数据...")
        
        # 首先检查mel和pinyin列表长度是否一致
        if len(self.mel_list) != len(self.pinyin_list):
            print(f"警告: mel列表长度({len(self.mel_list)}) != pinyin列表长度({len(self.pinyin_list)})")
            # 取较短的长度，避免索引越界
            min_length = min(len(self.mel_list), len(self.pinyin_list))
            self.mel_list = self.mel_list[:min_length]
            self.pinyin_list = self.pinyin_list[:min_length]
            print(f"已截断到相同长度: {min_length}")
        
        valid_indices = []
        skipped_count = 0
        
        for i in range(len(self.mel_list)):
            try:
                # 检查mel文件是否存在
                mel_path = self.mel_list[i]
                if not os.path.exists(mel_path):
                    print(f"跳过: mel文件不存在 {mel_path}")
                    skipped_count += 1
                    continue
                
                # 检查pinyin是否为空
                pinyin = self.pinyin_list[i]
                if not pinyin or len(pinyin) == 0:
                    print(f"跳过: 第{i}行pinyin为空")
                    skipped_count += 1
                    continue
                
                # 尝试加载mel并检查其基本属性
                try:
                    mel = torch.load(mel_path)
                    
                    # 检查mel是否为tensor
                    if not isinstance(mel, torch.Tensor):
                        print(f"跳过: 第{i}行mel不是tensor类型")
                        skipped_count += 1
                        continue
                    
                    # 检查mel形状是否合理
                    if len(mel.shape) < 2:
                        print(f"跳过: 第{i}行mel形状异常 {mel.shape}")
                        skipped_count += 1
                        continue
                    
                    # 检查mel长度
                    mel_length = mel.shape[-1] if len(mel.shape) >= 2 else mel.shape[0]
                    if mel_length > self.max_mel_length or mel_length < self.min_mel_length:
                        print(f"跳过: 第{i}行mel长度超出范围 {mel_length} (范围: {self.min_mel_length}-{self.max_mel_length})")
                        skipped_count += 1
                        continue
                    
                    # 检查mel是否包含异常值
                    if torch.isnan(mel).any() or torch.isinf(mel).any():
                        print(f"跳过: 第{i}行mel包含NaN或Inf")
                        skipped_count += 1
                        continue
                    
                    # 如果所有检查都通过，则保留这个索引
                    valid_indices.append(i)
                    
                except Exception as e:
                    print(f"跳过: 第{i}行mel加载失败 {mel_path}: {e}")
                    skipped_count += 1
                    continue
                    
            except Exception as e:
                print(f"跳过: 第{i}行处理失败: {e}")
                skipped_count += 1
                continue
        
        # 更新列表，只保留有效的数据
        self.mel_list = [self.mel_list[i] for i in valid_indices]
        self.pinyin_list = [self.pinyin_list[i] for i in valid_indices]
        
        print(f"过滤完成: 跳过了{skipped_count}个异常样本，保留{len(valid_indices)}个有效样本")

    def smart_padding(self, batch_data):
        
        # 分离mel和phoneme_ids
        mels = [item[0] for item in batch_data]
        phoneme_ids_list = [item[1] for item in batch_data]
        
        # 获取batch大小
        batch_size = len(batch_data)
        
        # 计算mel的最大长度和维度
        mel_lengths = [mel.shape[-1] for mel in mels]  # mel的时间维度通常是最后一维
        max_mel_length = max(mel_lengths)
        mel_dim = mels[0].shape[0] if len(mels[0].shape) == 2 else mels[0].shape[-2]  # mel特征维度
        
        # 计算phoneme的最大长度
        phoneme_lengths = [len(phoneme_ids) for phoneme_ids in phoneme_ids_list]
        max_phoneme_length = max(phoneme_lengths)
        
        # 初始化padded tensors
        padded_mels = torch.zeros(batch_size, mel_dim, max_mel_length, device=self.device)
        padded_phoneme_ids = torch.full((batch_size, max_phoneme_length), 0, 
                                       dtype=torch.long, device=self.device)  # 用0进行padding
        
        # 填充数据
        for i in range(batch_size):
            # 填充mel数据
            mel = mels[i]
            if len(mel.shape) == 2:  # [mel_dim, time_steps]
                padded_mels[i, :, :mel.shape[1]] = mel
            else:  # 如果是其他形状，需要调整
                padded_mels[i, :, :mel.shape[-1]] = mel
            
            # 填充phoneme数据
            phoneme_ids = phoneme_ids_list[i]
            if isinstance(phoneme_ids, torch.Tensor):
                padded_phoneme_ids[i, :len(phoneme_ids)] = phoneme_ids
            else:
                padded_phoneme_ids[i, :len(phoneme_ids)] = torch.tensor(phoneme_ids, 
                                                                       dtype=torch.long, 
                                                                       device=self.device)
        
        # 转换长度为tensor
        mel_lengths = torch.tensor(mel_lengths, dtype=torch.long, device=self.device)
        phoneme_lengths = torch.tensor(phoneme_lengths, dtype=torch.long, device=self.device)
        
        return padded_mels, padded_phoneme_ids, mel_lengths, phoneme_lengths

    def collate_fn(self, batch):
        return self.smart_padding(batch)
        # 返回的batch格式：
        # (padded_mels, padded_phoneme_ids, mel_lengths, phoneme_lengths)
        # 形状分别为：
        # padded_mels:        [batch_size, n_mels, max_mel_length]     例如: [2, 128, 690]
        # padded_phoneme_ids: [batch_size, max_phoneme_length]         例如: [2, 53]
        # mel_lengths:        [batch_size]                             例如: [690, 524]
        # phoneme_lengths:    [batch_size]                             例如: [53, 49]

if __name__ == "__main__":
    # 创建测试数据集
    # By Claude-4-sonnet
    try:
        # 测试不同的参数设置
        print("=== 创建数据集并测试过滤功能 ===")
        dataset = ShoreDataset("data/mel_list.list", "data/pinyin_list.list", 
                              max_mel_length=3000, min_mel_length=100)
        print(f"数据集总长度: {len(dataset)}")
        
        # 测试过滤功能是否正常工作
        print(f"\n=== 测试数据过滤效果 ===")
        print(f"最大mel长度限制: {dataset.max_mel_length}")
        print(f"最小mel长度限制: {dataset.min_mel_length}")
        
        # 测试前几个样本的原始数据
        print("\n=== 测试前3个样本的原始数据 ===")
        original_samples = []
        for i in range(min(3, len(dataset))):
            print(f"\n--- 样本 {i} ---")
            try:
                mel, phoneme_ids = dataset[i]
                original_samples.append((mel, phoneme_ids))
                
                print(f"mel文件路径: {dataset.mel_list[i]}")
                print(f"拼音列表: {dataset.pinyin_list[i]}")
                
                # 重点关注phoneme_ids的具体内容
                print(f"\n[PHONEME_IDS详细信息]")
                print(f"  phoneme_ids形状: {phoneme_ids.shape}")
                print(f"  phoneme_ids长度: {len(phoneme_ids)}")
                print(f"  phoneme_ids完整序列: {phoneme_ids.tolist()}")
                
                # 显示mel信息，特别关注长度是否在范围内
                print(f"\n[MEL详细信息]")
                print(f"  mel形状: {mel.shape}")
                mel_length = mel.shape[1] if len(mel.shape) == 2 else mel.shape[-1]
                print(f"  mel时间长度: {mel_length}")
                print(f"  mel长度是否在范围内: {dataset.min_mel_length} <= {mel_length} <= {dataset.max_mel_length}")
                print(f"  mel数值范围: [{mel.min().item():.4f}, {mel.max().item():.4f}]")
                
                # 检查mel/phoneme比例
                ratio = mel_length / len(phoneme_ids) if len(phoneme_ids) > 0 else 0
                print(f"  mel/phoneme比例: {ratio:.2f}")
                
            except Exception as e:
                print(f"处理样本 {i} 时出错: {e}")
                import traceback
                traceback.print_exc()
        
        # 重点测试smart_padding功能
        print(f"\n" + "="*60)
        print(f"=== 重点测试SMART_PADDING功能 ===")
        print(f"="*60)
        
        if len(original_samples) >= 2:
            # 取前两个样本进行batch测试
            test_batch = original_samples[:2]
            
            print(f"\n--- PADDING前的原始数据 ---")
            for i, (mel, phoneme_ids) in enumerate(test_batch):
                mel_length = mel.shape[1] if len(mel.shape) == 2 else mel.shape[-1]
                print(f"样本{i+1}:")
                print(f"  Mel形状: {mel.shape}, 时间长度: {mel_length}")
                print(f"  Phoneme长度: {len(phoneme_ids)}")
                print(f"  Phoneme完整序列: {phoneme_ids.tolist()}")
            
            # 应用smart_padding
            try:
                padded_mels, padded_phoneme_ids, mel_lengths, phoneme_lengths = dataset.smart_padding(test_batch)
                
                print(f"\n--- PADDING后的数据 ---")
                print(f"Padded mels形状: {padded_mels.shape}")
                print(f"Padded phoneme_ids形状: {padded_phoneme_ids.shape}")
                print(f"Mel实际长度: {mel_lengths.tolist()}")
                print(f"Phoneme实际长度: {phoneme_lengths.tolist()}")
                
                # 验证padding后的长度不会超过max_mel_length
                max_padded_mel_length = padded_mels.shape[2]
                print(f"Padding后最大mel长度: {max_padded_mel_length}")
                print(f"是否超过限制: {max_padded_mel_length > dataset.max_mel_length}")
                
                print(f"\n--- 详细的PADDING效果展示 ---")
                for i in range(len(test_batch)):
                    print(f"\n样本{i+1}的padding效果:")
                    original_length = phoneme_lengths[i].item()
                    padded_length = padded_phoneme_ids.shape[1]
                    
                    print(f"  原始phoneme长度: {original_length}")
                    print(f"  Padding后总长度: {padded_length}")
                    print(f"  需要padding的长度: {padded_length - original_length}")
                    
                    # 显示完整的phoneme序列
                    phoneme_sequence = padded_phoneme_ids[i].tolist()
                    print(f"  完整的phoneme序列: {phoneme_sequence}")
                    
                    # 分别显示原始部分和padding部分
                    original_part = phoneme_sequence[:original_length]
                    padding_part = phoneme_sequence[original_length:]
                    
                    print(f"  原始部分: {original_part}")
                    print(f"  Padding部分: {padding_part}")
                    
                    # 验证padding部分是否都是0
                    if padding_part:
                        all_0 = all(x == 0 for x in padding_part)
                        print(f"  Padding部分是否全为0: {all_0}")
                        if not all_0:
                            print(f"  ⚠️  警告: Padding部分包含非0的值!")
                    else:
                        print(f"  无需padding (已是最长序列)")
                
                # 验证batch中所有序列长度是否一致
                print(f"\n--- BATCH一致性验证 ---")
                all_same_length = all(len(seq) == padded_phoneme_ids.shape[1] for seq in padded_phoneme_ids)
                print(f"Batch中所有phoneme序列长度是否一致: {all_same_length}")
                print(f"统一的序列长度: {padded_phoneme_ids.shape[1]}")
                
                # 显示mel的padding效果
                print(f"\n--- MEL的PADDING效果 ---")
                for i in range(len(test_batch)):
                    original_mel_length = mel_lengths[i].item()
                    padded_mel_length = padded_mels.shape[2]
                    print(f"样本{i+1}: 原始mel长度={original_mel_length}, padding后长度={padded_mel_length}")
                
            except Exception as e:
                print(f"Smart padding测试失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 测试DataLoader兼容性
        print(f"\n" + "="*60)
        print(f"=== 测试DATALOADER兼容性 ===")
        print(f"="*60)
        try:
            from torch.utils.data import DataLoader
            
            # 创建DataLoader
            dataloader = DataLoader(dataset, batch_size=2, shuffle=False, 
                                  collate_fn=dataset.collate_fn)
            
            # 获取第一个batch
            batch_data = next(iter(dataloader))
            padded_mels, padded_phoneme_ids, mel_lengths, phoneme_lengths = batch_data
            
            print(f"DataLoader成功创建并获取batch!")
            print(f"Batch中的数据形状:")
            print(f"  Padded mels: {padded_mels.shape}")
            print(f"  Padded phoneme_ids: {padded_phoneme_ids.shape}")
            print(f"  Mel lengths: {mel_lengths.tolist()}")
            print(f"  Phoneme lengths: {phoneme_lengths.tolist()}")
            
            # 验证所有mel长度都在限制范围内
            max_mel_in_batch = max(mel_lengths.tolist())
            min_mel_in_batch = min(mel_lengths.tolist())
            print(f"  Batch中mel长度范围: [{min_mel_in_batch}, {max_mel_in_batch}]")
            print(f"  是否都在限制范围内: {min_mel_in_batch >= dataset.min_mel_length and max_mel_in_batch <= dataset.max_mel_length}")
            
            # 显示DataLoader中的padding效果
            print(f"\nDataLoader中的padding效果:")
            for i in range(padded_phoneme_ids.shape[0]):
                phoneme_seq = padded_phoneme_ids[i].tolist()
                actual_length = phoneme_lengths[i].item()
                print(f"  样本{i+1}: 实际长度={actual_length}")
                print(f"    完整序列: {phoneme_seq}")
                print(f"    原始部分: {phoneme_seq[:actual_length]}")
                print(f"    Padding部分: {phoneme_seq[actual_length:]}")
            
        except Exception as e:
            print(f"DataLoader测试失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 测试拼音到ID的转换功能
        print(f"\n" + "="*60)
        print(f"=== 测试拼音转换功能 ===")
        print(f"="*60)
        test_pinyin = ['ni3', 'hao3', 'ma']
        test_ids = pinyin_to_ids(test_pinyin)
        print(f"测试拼音: {test_pinyin}")
        print(f"转换结果: {test_ids}")
        
        # 转换为tensor后的结果
        test_tensor = torch.tensor(test_ids, dtype=torch.long)
        print(f"转换为tensor: {test_tensor.tolist()}")
        print(f"tensor形状: {test_tensor.shape}")
        
    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
        print("请确保 data/mel_list.list 和 data/pinyin_list.list 文件存在")
        
    except Exception as e:
        print(f"初始化数据集时出错: {e}")
        import traceback
        traceback.print_exc()
    
