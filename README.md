# Shore-TTS

基于 **BN-MDCT** 特征与 **CFM + DiT** 主干的非自回归语音合成系统。

Shore-TTS 的核心思路是：不依赖 vocoder 也不依赖音频 codec，而是让神经网络直接预测一种可由传统信号处理算法**无损复原**的声学特征——BN-MDCT 谱，从而从根本上消除 vocoder/codec 引入的音质损失。当前实现以 F5-TTS 为蓝本，将 mel 谱替换为 BN-MDCT，使用 Conditional Flow Matching 训练 DiT。

## 项目结构

```
Shore-TTS/
├── shore_tts/
│   ├── configs/
│   │   ├── mdct.json              # BN-MDCT 特征配置
│   │   └── pretrain.json          # 预训练超参数配置
│   ├── datasets/
│   │   └── dataset.py             # WebDataset 数据管线 & 动态组批
│   ├── models/
│   │   ├── cfm.py                 # Conditional Flow Matching 模型
│   │   ├── dit.py                 # DiT 主干（文本嵌入 + 输入嵌入 + Transformer 块）
│   │   ├── modules.py             # 基础模块（MDCTSpec、DiTBlock、AdaLN、ConvNeXtV2 等）
│   │   └── utils.py               # 工具函数（RoPE、mask、EPSS 采样等）
│   ├── optimizer/
│   │   ├── __init__.py            # 优化器导出
│   │   ├── muon.py                # Muon 优化器 & Muon_AdamW 组合优化器
│   │   └── chained_optimizer.py   # 多优化器链式调度框架
│   ├── text/
│   │   └── tokenizer.py           # 拼音分词器，支持多音字 & 词表训练
│   ├── utils/
│   │   ├── build.py               # 模型/优化器/调度器/数据加载器构建
│   │   ├── loss.py                # 频率加权 MSE 损失（FrequencyWeightedMSELoss）
│   │   ├── spectrogram.py         # BN-MDCT 特征提取与可逆重建核心实现
│   │   └── trainer.py             # 训练器（Accelerate DDP、EMA、TensorBoard、断点续训）
│   └── train.py                   # 训练入口
├── tools/
│   ├── inference.py               # 推理 & voice cloning 脚本
│   └── pack.py                    # 多卷 tar 归档 → 标准 WebDataset 分片转换
├── checkpoints/
│   └── Shore-TTS-0.1/            # 预训练检查点（v0.1）
│       ├── config.json
│       ├── vocab.json
│       └── model.pth
└── assets/                        # 参考资源 & F5-TTS 源码
```

## 依赖

- Python >= 3.10
- PyTorch >= 2.0（推荐 2.4+，支持原生 RMSNorm）
- [flash-attn](https://github.com/Dao-AILab/flash-attention)（可选，推荐；不支持的平台自动回退到 PyTorch SDPA）
- accelerate
- ema_pytorch
- torchaudio
- webdataset
- rjieba, pypinyin（拼音分词）
- einops
- torchdiffeq
- tensorboard

## 已实现

- 训练入口 `shore_tts/train.py`，支持 DDP 多卡、EMA、混合精度、梯度累积、断点续训
- CFM + DiT 模型主干（`shore_tts/models/cfm.py`、`dit.py`、`modules.py`）
- BN-MDCT 特征提取与可逆重建（`shore_tts/utils/spectrogram.py`），当前配置 `sample_rate=24000 / hop_length=100 / n_bands=20`，特征维度 `120`
- WebDataset 数据管线（`shore_tts/datasets/dataset.py`），在线解码 tar 分片并即时提取 MDCT，支持动态组批与按分卷大小均衡分配
- 拼音分词器（`shore_tts/text/tokenizer.py`），支持多音字，可从 tar 分片训练词表
- 推理与 voice cloning（`tools/inference.py`），支持参考音频 few-shot、CFG、sway sampling、语速控制、固定时长
- 频率加权 MSE 损失（`shore_tts/utils/loss.py`），低频权重更高，alpha 参数控制衰减程度
- Muon 优化器（`shore_tts/optimizer/muon.py`），使用 Newton-Schulz 正交化 + 动量，2D+ 参数走 Muon，其余走 AdamW
- 梯度检查点 / 激活重计算（`checkpoint_activations`），降低长序列显存占用
- 检查点管理：自动轮转、完整目录格式（config + vocab + weights）、断点续训、`model_last.pt` 轻量保存
- TensorBoard 日志 & 训练中自动采样生成
- Flash Attention 支持（`flash_attn` 后端），自动回退到 PyTorch SDPA

> **Note for AI Coding Agents：** 下文"整体流程 / 后续研究"包含尚未实现或尚未定型的方案，编码时请以代码现状为准，优先核对真实入口、配置和现有模块。

# 整体流程

## 1. 数据准备

原始数据为 **WebDataset** 格式的 tar 分片，每个 tar 中包含配对的音频文件（`.opus`/`.wav`/`.flac` 等）与对应文本（`.txt` 或 `.json`）。使用 `tools/pack.py` 可将多卷 tar 归档转换为标准 WebDataset 分片：

```bash
python tools/pack.py -i /path/to/multi-volume/tars -o shards/ -s 10000
```

## 2. 词表构建

从 tar 分片中提取所有文本，构建拼音词表：

```bash
python -m shore_tts.text.tokenizer --data-path shards/ --output checkpoints/vocab/vocab.json
```

可选参数：
- `--text-suffix .txt`：指定 tar 内文本文件后缀（默认 `.txt`）
- `--disable-polyphone`：关闭多音字感知的拼音转换

## 3. 特征提取（BN-MDCT）

训练时在线提取，无需离线预处理。核心特征为 **BN-MDCT（Band-Normalized MDCT）**，定义于 `shore_tts/utils/spectrogram.py`：

1. 对音频做 MDCT 变换（`hop_length=100, n_fft=200`），得到 100 维频谱系数
2. 将频带划分为 20 个子带，计算各子带能量包络
3. 用能量包络对系数做归一化，拼接 log 能量 + 归一化系数 → **120 维特征**
4. 逆过程可**无损还原**波形（MDCT 本身是完美重构变换）

配置见 `shore_tts/configs/mdct.json`。

## 4. 文本处理

中文文本经 `rjieba` 分词后由 `pypinyin` 转为带声调拼音（`shore_tts/text/tokenizer.py`），如"你好"→`ni3 hao3`；非中文字符原样保留。词表（`vocab.json`）在训练时构建，涵盖拼音音节、ASCII 字符及多语种字符。

## 5. 训练

入口：`shore_tts/train.py`，配置：`shore_tts/configs/pretrain.json`。

```bash
python shore_tts/train.py --config shore_tts/configs/pretrain.json
```

**模型架构**：CFM（Conditional Flow Matching）+ DiT（Diffusion Transformer）

```
文本 ──→ 拼音嵌入 + ConvNeXtV2 ──→ 文本编码
                                          ┐
噪声 BN-MDCT + 条件 BN-MDCT + 文本编码 ──→ DiT (×22 层) ──→ 预测流场
                                          ┘
      时间步嵌入 (AdaLN 调制)    旋转位置编码 (RoPE)
```

**训练目标**：对目标 BN-MDCT 特征施加随机掩码（span masking），在噪声与目标之间线性插值 `φ_t = (1-t)·ε + t·x`，让模型预测流场 `v = x - ε`，仅在掩码区域计算**频率加权 MSE 损失**（低频权重更高）。

**CFG 训练**：以 0.3 概率丢弃音频条件、0.2 概率同时丢弃音频与文本条件，用于推理时 Classifier-Free Guidance。

**优化器**：默认使用 **Muon_AdamW** 组合优化器——2D 及以上参数（如权重矩阵）走 Muon（Newton-Schulz 正交化 + Nesterov 动量），其余参数（Embedding、偏置等）走 AdamW。也可通过配置 `optimizer_type: "adamw"` 回退到纯 AdamW。

**学习率调度**：Linear Warmup + Linear Decay，`warmup_steps` 控制预热步数，`final_lr_scale` 控制衰减终点。

**梯度检查点**：启用 `checkpoint_activations: true` 后，DiT 前向使用 `torch.utils.checkpoint` 重计算，用计算换显存，适合长序列训练。

## 6. 推理

```bash
# 基础推理
python tools/inference.py \
    --checkpoint checkpoints/Shore-TTS-0.1 \
    --text "你好世界" \
    --output output.wav

# Voice cloning（使用参考音频）
python tools/inference.py \
    --checkpoint checkpoints/Shore-TTS-0.1 \
    --text "你好世界" \
    --ref_audio ref.wav \
    --ref_text "参考文本" \
    --output output.wav
```

推理参数：
- `--steps`：ODE 求解步数（默认 32）
- `--cfg_strength`：CFG 强度（默认 1.0）
- `--sway_sampling_coef`：Sway sampling 系数（默认关闭）
- `--speed`：语速因子，越大越快（默认 1.0）
- `--fix_duration`：固定输出时长（秒），覆盖比例估算
- `--seed`：随机种子

# TO DO LIST

- 去除 F5 的低效实现，参考 F5 的思路，但是提高里面的各种效率，加快收敛速度
- 加入 HiFi-GAN 里面的 MPD 作为判别器，提升模型音质效果
- 引入 wvmos 或者其他的自动 mos 评估模型，用于可视化训练过程中模型的基本表现