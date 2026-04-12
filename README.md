# Shore-TTS

基于 **BN-MDCT** 特征与 **CFM + DiT** 主干的非自回归语音合成系统。

Shore-TTS 的核心思路是：不依赖 vocoder 也不依赖音频 codec，而是让神经网络直接预测一种可由传统信号处理算法**无损复原**的声学特征——BN-MDCT 谱，从而从根本上消除 vocoder/codec 引入的音质损失。当前实现以 F5-TTS 为蓝本，将 mel 谱替换为 BN-MDCT，使用 Conditional Flow Matching 训练 DiT。

目前已经跑通了！也可以生成音频了！唯一的缺陷就是音质和表现力不够强，一旦这两个够了，对于语音来说也够一个 ICML Oral 了。

**已实现：**
- 训练入口 `shore_tts/train.py`，支持 DDP 多卡、EMA、混合精度、断点续训
- CFM + DiT 模型主干（`shore_tts/models/cfm.py`、`dit.py`、`modules.py`）
- BN-MDCT 特征提取与可逆重建（`shore_tts/utils/spectrogram.py`），当前配置 `sample_rate=24000 / hop_length=100 / n_bands=20`，特征维度 `120`
- WebDataset 数据管线（`shore_tts/datasets/dataset.py`），在线解码 tar 分片并即时提取 MDCT
- 拼音分词器（`shore_tts/text/tokenizer.py`），支持多音字
- 推理与 voice cloning（`tools/inference.py`），支持参考音频 few-shot、CFG、sway sampling

> **Note for AI Coding Agents：** 下文”整体流程 / 后续研究”包含尚未实现或尚未定型的方案，编码时请以代码现状为准，优先核对真实入口、配置和现有模块。

# 整体流程

## 1. 数据准备

原始数据为 **WebDataset** 格式的 tar 分片，每个 tar 中包含配对的音频文件（`.opus`/`.wav`/`.flac` 等）与对应文本（`.txt`）。使用 `tools/pack.py` 可将多卷 tar 归档转换为标准 WebDataset 分片：

```
python tools/pack.py --input data.tar.* --output_dir shards/ --samples_per_shard 10000
```

## 2. 特征提取（BN-MDCT）

训练时在线提取，无需离线预处理。核心特征为 **BN-MDCT（Band-Normalized MDCT）**，定义于 `shore_tts/utils/spectrogram.py`：

1. 对音频做 MDCT 变换（`hop_length=100, n_fft=200`），得到 100 维频谱系数
2. 将频带划分为 20 个子带，计算各子带能量包络
3. 用能量包络对系数做归一化，拼接 log 能量 + 归一化系数 → **120 维特征**
4. 逆过程可**无损还原**波形（MDCT 本身是完美重构变换）

配置见 `shore_tts/configs/mdct.json`。

## 3. 文本处理

中文文本经 `rjieba` 分词后由 `pypinyin` 转为带声调拼音（`shore_tts/text/tokenizer.py`），如"你好"→`ni3 hao3`；非中文字符原样保留。词表（`vocab.json`，2625 token）在训练时构建，涵盖拼音音节、ASCII 字符及多语种字符。

## 4. 训练

入口：`shore_tts/train.py`，配置：`shore_tts/configs/pretrain.json`。

**模型架构**：CFM（Conditional Flow Matching）+ DiT（Diffusion Transformer）

```
文本 ──→ 拼音嵌入 + ConvNeXtV2 ──→ 文本编码
                                          ┐
噪声 BN-MDCT + 条件 BN-MDCT + 文本编码 ──→ DiT (×22 层) ──→ 预测流场
                                          ┘
      时间步嵌入 (AdaLN 调制)
```

**训练目标**：对目标 BN-MDCT 特征施加随机掩码（span masking），在噪声与目标之间线性插值 `φ_t = (1-t)·ε + t·x`，让模型预测流场 `v = x - ε`，仅在掩码区域计算 MSE 损失。

**CFG 训练**：以 0.3 概率丢弃音频条件、0.2 概率同时丢弃音频与文本条件，用于推理时 Classifier-Free Guidance。

**训练设置**：AdamW (lr=2.5e-4) + 线性 warmup 1000 步 + 线性衰减、EMA (0.9999)、混合精度 fp16、DDP 多卡、梯度裁剪 (max_norm=1.0)、每 2000 步存检查点（保留最近 10 个）。

## 5. 推理

入口：`tools/inference.py`。

```
参考音频 ──→ BN-MDCT 编码 ──→ 条件特征
参考文本 + 目标文本 ──→ 拼音序列 ──→ 文本编码
                                          ┐
随机噪声 ──────────────────────────────→ ODE 求解 (32步) ──→ 生成 BN-MDCT
                                          ┘
      条件特征 + 文本编码 + CFG
```

1. 加载检查点（`config.json` + `vocab.json` + `model.pth`）
2. 参考音频 → BN-MDCT 特征作为条件
3. 根据参考音频时长估算目标时长（约每拼音 6 帧 MDCT）
4. ODE 采样（默认 32 步），支持 CFG、sway sampling、EPSS 加速
5. 截取生成部分，逆 BN-MDCT 还原为波形，保存 WAV


# TO DO LIST

- 去除F5的低效实现，参考F5的思路，但是提高里面的各种效率，加快收敛速度。
- 加muon作为训练的优化器，去除AdamW的低效训练速度。
- 加入hifigan里面的MPD作为判别器，提升模型音质效果。