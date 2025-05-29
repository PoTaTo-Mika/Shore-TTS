
import librosa
import soundfile as sf
import torch
import os
import sys

# 设置根目录，确保能正确导入shore_tts模块
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)  # 获取项目根目录
sys.path.insert(0, root_dir)

from shore_tts.modules.vocoder import ADaMoSHiFiGANV1 as Hifigan

device = 'cuda'
# 使用相对于根目录的路径
model_path = os.path.join(root_dir, 'checkpoints/vocoder/model.safetensors')
model = Hifigan(model_path).to(device)

def extract_one_file(wav_path, output_dir):
    # 获取wav名称
    wav_name = os.path.basename(wav_path)
    # 加载音频
    wav, sr = librosa.load(wav_path, sr=44100, mono=True)
    # 转换为tensor
    wav = torch.from_numpy(wav).float()[None].to(device)
    # 编码为mel
    mel = model.encode(wav)
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    # 保存mel，文件名与wav一致
    output_path = os.path.join(output_dir, wav_name.replace('.wav', '.pt'))
    torch.save(mel, output_path)
    print(f"已保存mel特征到: {output_path}")

if __name__ == "__main__":
    # 使用根目录下的test.wav文件
    extract_one_file('data/wav/test2.wav', './')
