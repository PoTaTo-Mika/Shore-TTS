# Shore-TTS

非自回归 TTS：BN-MDCT 特征 + CFM + DiT。无 vocoder/codec，网络直接预测 BN-MDCT 谱（可由信号处理算法无损还原波形）。

你可以根据这个markdown获取大部分信息，如果还有不懂的可以查阅更加详细的Human documents.

## 关键路径

| 用途 | 文件 |
|------|------|
| 训练入口 | `shore_tts/train.py` |
| 训练配置 | `shore_tts/configs/pretrain.json` |
| MDCT 配置 | `shore_tts/configs/mdct.json` |
| CFM 模型 | `shore_tts/models/cfm.py` |
| DiT 主干 | `shore_tts/models/dit.py` |
| 基础模块 | `shore_tts/models/modules.py` |
| BN-MDCT 特征提取/重建 | `shore_tts/utils/spectrogram.py` |
| 频率加权 MSE 损失 | `shore_tts/utils/loss.py` |
| 训练器 | `shore_tts/utils/trainer.py` |
| 拼音分词器 | `shore_tts/text/tokenizer.py` |
| 数据管线 | `shore_tts/datasets/dataset.py` |
| Muon 优化器 | `shore_tts/optimizer/muon.py` |
| 推理 | `tools/inference.py` |
| 打包分片 | `tools/pack.py` |

## 环境安装

```bash
# 默认新建了一个虚拟环境
pip install -r requirements.txt
```

## 一步式命令

复现训练：
```bash
# 下载Emilia数据集到某个位置，只保留ZH和EN两个子集
accelerate launch shore_tts/train.py --config shore_tts/configs/pretrain.json
```

## 架构要点

- **BN-MDCT**：MDCT(hop=100, n_fft=200) → 20 子带归一化 → 拼接 log 能量 + 归一化系数 = **120 维**特征，可无损还原波形
- **模型**：CFM + DiT×22 层，AdaLN 调制 + RoPE；文本经拼音嵌入 + ConvNeXtV2 编码
- **训练目标**：span masking 掩码区域计算频率加权 MSE；线性插值 φ_t = (1-t)·ε + t·x，预测流场 v = x - ε
- **CFG 训练**：0.3 概率丢音频条件，0.2 概率同时丢音频+文本
- **优化器**：Muon_AdamW（2D+ 参数走 Muon，其余走 AdamW）；可配 `optimizer_type: "adamw"` 回退纯 AdamW
- **调度**：Linear Warmup + Linear Decay
- **梯度检查点**：`checkpoint_activations: true` 用计算换显存
- **数据**：WebDataset tar 分片，在线提取 MDCT，动态组批
- **分词**：rjieba 分词 + pypinyin 转带调拼音，支持多音字

## 注意

- 代码现状为准，"后续研究"仅作参考
- TF32 会使训练变慢，不要开启
- 检查点目录格式：config.json + vocab.json + model.pth
