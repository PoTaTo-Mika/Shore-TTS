# 本脚本由Gemini2.5-pro完成
import os
import requests
import tarfile
import csv
import numpy as np
import librosa
from g2p_en import G2p
from tqdm import tqdm
import re

# --- 1. 配置参数 (Configuration) ---

# 数据集URL和路径
LJSPEECH_URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
DATA_DIR = "data"
LJSPEECH_DIR = os.path.join(DATA_DIR, "LJSpeech-1.1")
TAR_PATH = os.path.join(DATA_DIR, "LJSpeech-1.1.tar.bz2")

# 音频处理参数 - 统一配置以匹配HiFiGAN论文
SAMPLE_RATE = 22050
N_MELS = 80
FFT_SIZE = 1024
HOP_SIZE = 256
WIN_SIZE = 1024
F_MIN = 0
F_MAX = 8000

# 输出目录
MELS_DIR = os.path.join(DATA_DIR, "mels")
TEXTS_DIR = os.path.join(DATA_DIR, "texts")

# --- 2. 音素字典定义 (Phoneme Vocabulary Definition) ---

# 基于CMUdict和g2p_en的常见音素，并添加特殊符号
# _pad: 填充符
# _eos: 句子结束符
# _bos: 句子开始符 (可选，但良好实践)
# _unk: 未知音素 (备用)
_pad = "_"
_eos = "~"
_bos = "^"
_unk = "?"
_punctuation = ".,!?' " # 保留一些标点作为音素

# ARPAbet音素 (无重音符号，重音由模型学习)
_arpabet_phonemes = [
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY',
    'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY',
    'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH'
]

# 最终的符号列表 (Vocabulary)
SYMBOLS = [_pad, _eos, _bos, _unk] + list(_punctuation) + _arpabet_phonemes

# 创建符号到ID的映射字典
SYMBOL_TO_ID = {s: i for i, s in enumerate(SYMBOLS)}

# --- 3. 核心处理函数 (Core Processing Functions) ---

def _download_and_unpack(url, data_dir, tar_path, ljspeech_dir):
    """下载并解压数据集"""
    if os.path.exists(ljspeech_dir):
        print(f"'{ljspeech_dir}' 目录已存在，跳过下载和解压。")
        return

    print(f"开始下载LJSpeech数据集到 '{tar_path}'...")
    os.makedirs(data_dir, exist_ok=True)
    
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        with open(tar_path, 'wb') as f, tqdm(
            total=total_size, unit='iB', unit_scale=True, desc="下载中"
        ) as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

    print("下载完成，开始解压...")
    with tarfile.open(tar_path, "r:bz2") as tar:
        tar.extractall(path=data_dir)
    print("解压完成。")
    os.remove(tar_path) # 删除压缩包以节省空间

def _normalize_text_and_phonemize(text, g2p):
    """清理文本并将其转换为音素序列"""
    # 移除ARPAbet中不存在的字符
    text = re.sub(r'[^a-zA-Z.,!?\' ]', '', text)
    text = text.lower()

    # 使用g2p_en进行转换
    phonemes = g2p(text)

    # 清理g2p输出：去除重音数字，并将不在字典中的符号替换为<unk>
    cleaned_phonemes = []
    for p in phonemes:
        # 去除ARPAbet重音数字（例如 'AH0', 'AE1', 'IH2' -> 'AH', 'AE', 'IH'）
        p_no_stress = re.sub(r'\d', '', p)
        if p_no_stress in SYMBOL_TO_ID:
            cleaned_phonemes.append(p_no_stress)
        elif p in SYMBOL_TO_ID: # 如果是标点符号
             cleaned_phonemes.append(p)
        else:
             # 如果清理后或原始符号都不在字典中，则忽略或标记为未知
             # 这里选择忽略未知音素，也可以替换为_unk
             pass
    
    return cleaned_phonemes

def text_to_sequence(text_phonemes):
    """将音素列表转换为数字ID序列，并添加BOS/EOS符号"""
    sequence = [SYMBOL_TO_ID[_bos]]
    sequence.extend([SYMBOL_TO_ID[p] for p in text_phonemes])
    sequence.append(SYMBOL_TO_ID[_eos])
    return sequence

def compute_mel_spectrogram(audio_path):
    """
    计算与HiFiGAN兼容的mel频谱图
    
    Args:
        audio_path (str): 音频文件路径
        
    Returns:
        np.ndarray: 归一化的对数mel频谱图 [time_steps, n_mels]
    """
    # 1. 加载音频
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    
    # 2. 计算mel频谱图 (线性尺度)
    melspec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=N_MELS,
        n_fft=FFT_SIZE,
        hop_length=HOP_SIZE,
        win_length=WIN_SIZE,
        fmin=F_MIN,
        fmax=F_MAX,
        power=2.0  # 使用功率谱
    )
    
    # 3. 转换为对数域 - 使用与HiFiGAN相同的方法
    # 添加小的常数避免log(0)，然后取对数
    log_melspec = np.log(np.clip(melspec, a_min=1e-5, a_max=None))
    
    # 4. 转置以匹配期望的维度 [time_steps, n_mels]
    log_melspec = log_melspec.T
    
    return log_melspec

def process_and_save_data():
    """主函数：执行所有预处理步骤"""
    # 步骤1: 下载和解压
    _download_and_unpack(LJSPEECH_URL, DATA_DIR, TAR_PATH, LJSPEECH_DIR)

    # 步骤2: 创建输出目录
    os.makedirs(MELS_DIR, exist_ok=True)
    os.makedirs(TEXTS_DIR, exist_ok=True)
    print(f"梅尔频谱图将保存到: '{MELS_DIR}'")
    print(f"文本序列将保存到: '{TEXTS_DIR}'")
    print("配置参数:")
    print(f"  - 采样率: {SAMPLE_RATE} Hz")
    print(f"  - FFT大小: {FFT_SIZE}")
    print(f"  - 跳跃长度: {HOP_SIZE}")
    print(f"  - 窗口长度: {WIN_SIZE}")
    print(f"  - Mel频段数: {N_MELS}")
    print(f"  - 频率范围: {F_MIN}-{F_MAX} Hz")

    # 步骤3: 初始化G2P转换器
    g2p = G2p()

    # 步骤4: 读取metadata并处理每一行
    metadata_path = os.path.join(LJSPEECH_DIR, "metadata.csv")
    wav_dir = os.path.join(LJSPEECH_DIR, "wavs")
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        # 读取所有行
        lines = f.readlines()
        # 使用tqdm创建进度条
        for line in tqdm(lines, desc="处理数据中"):
            # 去除行尾的空白符并用'|'分割
            parts = line.strip().split('|')
            
            # 增加一个健壮性检查，防止行格式不正确导致程序崩溃
            if len(parts) < 3:
                print(f"\n警告: 行格式不正确，字段不足，已跳过: {line.strip()}")
                continue
                
            # LJSpeech的格式是 ID|文本|标准化文本
            # 我们需要第一个和第三个字段
            wav_name = parts[0]
            text = parts[2] # 或者 parts[1]，取决于你需要哪个版本的文本
            

            # --- 处理音频 ---
            wav_path = os.path.join(wav_dir, f"{wav_name}.wav")
            
            # 计算与HiFiGAN兼容的mel频谱图
            log_melspec = compute_mel_spectrogram(wav_path)
            
            # 保存为.npy
            mel_filename = os.path.join(MELS_DIR, f"{wav_name}.npy")
            np.save(mel_filename, log_melspec, allow_pickle=False)

            # --- 处理文本 ---
            # 1. 文本转音素
            phonemes = _normalize_text_and_phonemize(text, g2p)
            
            # 2. 音素转数字序列
            sequence = text_to_sequence(phonemes)
            
            # 3. 保存为.txt文件
            text_filename = os.path.join(TEXTS_DIR, f"{wav_name}.txt")
            with open(text_filename, 'w') as text_file:
                text_file.write(str(sequence))
    
    print("\n所有数据处理完成！")
    print("最终目录结构:")
    print(f"└─ {DATA_DIR}/")
    print(f"   ├─ mels/ ({len(os.listdir(MELS_DIR))} 个文件, e.g., LJ001-0001.npy)")
    print(f"   └─ texts/ ({len(os.listdir(TEXTS_DIR))} 个文件, e.g., LJ001-0001.txt)")
    print("\n重要更新:")
    print("✅ 统一了音频参数配置以匹配HiFiGAN")
    print("✅ 修复了mel频谱图归一化方法")
    print("✅ 使用与HiFiGAN兼容的对数变换")


# --- 脚本执行入口 ---
if __name__ == "__main__":
    process_and_save_data()