import os
import sys
import librosa
import soundfile as sf
import torch
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import time

# 设置根目录，确保能正确导入shore_tts模块
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))  # 获取项目根目录
sys.path.insert(0, root_dir)

from shore_tts.modules.vocoder import ADaMoSHiFiGANV1 as Hifigan

class MultiGPUMelProcessor:
    """多GPU mel特征处理器"""
    
    def __init__(self, model_path, output_dir):
        self.model_path = model_path
        self.output_dir = output_dir
        self.gpu_count = torch.cuda.device_count()
        self.models = {}
        self.lock = threading.Lock()
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"检测到 {self.gpu_count} 个GPU设备")
        
    def load_model_for_gpu(self, gpu_id):
        """为指定GPU加载模型"""
        device = f'cuda:{gpu_id}'
        try:
            model = Hifigan(self.model_path).to(device)
            with self.lock:
                self.models[gpu_id] = model
            print(f"✓ GPU {gpu_id} 模型加载完成")
            return True
        except Exception as e:
            print(f"✗ GPU {gpu_id} 模型加载失败: {e}")
            return False
    
    def process_file_on_gpu(self, wav_path, gpu_id):
        """在指定GPU上处理单个文件"""
        try:
            device = f'cuda:{gpu_id}'
            model = self.models[gpu_id]
            
            # 获取wav文件名（不含扩展名）
            wav_name = os.path.splitext(os.path.basename(wav_path))[0]
            
            # 加载音频
            wav, sr = librosa.load(wav_path, sr=44100, mono=True)
            
            # 转换为tensor
            wav_tensor = torch.from_numpy(wav).float()[None].to(device)
            
            # 编码为mel
            with torch.no_grad():
                mel = model.encode(wav_tensor)
            
            # 保存mel特征
            output_path = os.path.join(self.output_dir, f"{wav_name}.pt")
            torch.save(mel.cpu(), output_path)
            
            return True, wav_name, None
            
        except Exception as e:
            return False, os.path.basename(wav_path), str(e)

def batch_process_mel(input_dir, output_dir, max_workers=None):
    """
    递归遍历输入目录，使用多GPU并行处理所有.wav文件并转换为mel特征
    
    Args:
        input_dir (str): 输入目录路径
        output_dir (str): 输出目录路径
        max_workers (int): 最大工作线程数，默认为GPU数量
    """
    # 检查模型文件
    model_path = os.path.join(root_dir, 'checkpoints/vocoder/model.safetensors')
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        return
    
    # 检查输入目录
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"错误: 输入目录不存在: {input_dir}")
        return
    
    # 递归查找所有.wav文件
    wav_files = list(input_path.rglob("*.wav"))
    if not wav_files:
        print(f"在目录 {input_dir} 中未找到任何.wav文件")
        return
    
    print(f"找到 {len(wav_files)} 个.wav文件")
    
    # 检查GPU可用性
    if not torch.cuda.is_available():
        print("CUDA不可用，使用CPU单线程处理...")
        return batch_process_mel_cpu(input_dir, output_dir)
    
    gpu_count = torch.cuda.device_count()
    if max_workers is None:
        max_workers = gpu_count
    
    print(f"使用 {gpu_count} 个GPU，{max_workers} 个工作线程进行并行处理...")
    
    # 初始化多GPU处理器
    processor = MultiGPUMelProcessor(model_path, output_dir)
    
    # 为每个GPU加载模型
    print("正在为各GPU加载模型...")
    load_success = []
    for gpu_id in range(gpu_count):
        success = processor.load_model_for_gpu(gpu_id)
        load_success.append(success)
    
    if not any(load_success):
        print("所有GPU模型加载失败，退出处理")
        return
    
    available_gpus = [i for i, success in enumerate(load_success) if success]
    print(f"成功加载模型的GPU: {available_gpus}")
    
    # 开始计时
    start_time = time.time()
    
    # 使用线程池进行并行处理
    success_count = 0
    error_count = 0
    
    def process_batch(files_batch, gpu_id):
        """处理一批文件"""
        batch_success = 0
        batch_errors = 0
        
        for wav_file in files_batch:
            success, filename, error = processor.process_file_on_gpu(str(wav_file), gpu_id)
            if success:
                batch_success += 1
                print(f"[GPU{gpu_id}] ✓ {filename}")
            else:
                batch_errors += 1
                print(f"[GPU{gpu_id}] ✗ {filename}: {error}")
        
        return batch_success, batch_errors
    
    # 将文件分配给不同的GPU
    files_per_gpu = len(wav_files) // len(available_gpus)
    remainder = len(wav_files) % len(available_gpus)
    
    with ThreadPoolExecutor(max_workers=len(available_gpus)) as executor:
        futures = []
        start_idx = 0
        
        for i, gpu_id in enumerate(available_gpus):
            # 计算当前GPU要处理的文件数量
            current_batch_size = files_per_gpu + (1 if i < remainder else 0)
            end_idx = start_idx + current_batch_size
            
            # 分配文件给当前GPU
            files_batch = wav_files[start_idx:end_idx]
            
            if files_batch:  # 确保有文件要处理
                future = executor.submit(process_batch, files_batch, gpu_id)
                futures.append(future)
                print(f"GPU {gpu_id} 分配到 {len(files_batch)} 个文件")
            
            start_idx = end_idx
        
        # 收集结果
        for future in as_completed(futures):
            batch_success, batch_errors = future.result()
            success_count += batch_success
            error_count += batch_errors
    
    # 计算处理时间
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"\n{'='*60}")
    print(f"处理完成!")
    print(f"成功: {success_count}, 失败: {error_count}")
    print(f"总处理时间: {processing_time:.2f}秒")
    print(f"平均每文件: {processing_time/len(wav_files):.3f}秒")
    print(f"处理速度: {len(wav_files)/processing_time:.2f} 文件/秒")
    print(f"{'='*60}")

def batch_process_mel_cpu(input_dir, output_dir):
    """CPU单线程处理（备用方案）"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化模型
    device = 'cpu'
    model_path = os.path.join(root_dir, 'checkpoints/vocoder/model.safetensors')
    
    try:
        model = Hifigan(model_path).to(device)
        print(f"模型已加载到设备: {device}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return
    
    # 使用pathlib递归遍历目录
    input_path = Path(input_dir)
    wav_files = list(input_path.rglob("*.wav"))
    
    print(f"开始CPU单线程处理...")
    start_time = time.time()
    
    success_count = 0
    error_count = 0
    
    for i, wav_file in enumerate(wav_files, 1):
        try:
            # 处理单个wav文件
            extract_mel_from_wav(str(wav_file), output_dir, model, device)
            success_count += 1
            print(f"[{i:3d}/{len(wav_files)}] ✓ {wav_file.name}")
        except Exception as e:
            error_count += 1
            print(f"[{i:3d}/{len(wav_files)}] ✗ {wav_file.name}: {e}")
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"\n处理完成! 成功: {success_count}, 失败: {error_count}")
    print(f"总处理时间: {processing_time:.2f}秒")

def extract_mel_from_wav(wav_path, output_dir, model, device):
    """
    从单个wav文件提取mel特征
    
    Args:
        wav_path (str): wav文件路径
        output_dir (str): 输出目录
        model: 已加载的模型
        device: 设备类型
    """
    # 获取wav文件名（不含扩展名）
    wav_name = os.path.splitext(os.path.basename(wav_path))[0]
    
    # 加载音频
    wav, sr = librosa.load(wav_path, sr=44100, mono=True)
    
    # 转换为tensor
    wav_tensor = torch.from_numpy(wav).float()[None].to(device)
    
    # 编码为mel
    with torch.no_grad():
        mel = model.encode(wav_tensor)
    
    # 保存mel特征
    output_path = os.path.join(output_dir, f"{wav_name}.pt")
    torch.save(mel.cpu(), output_path)

if __name__ == "__main__":
    # 示例用法
    input_directory = "data/wav"  # 输入目录
    output_directory = "data/mel"  # 输出目录
    
    batch_process_mel(input_directory, output_directory)
