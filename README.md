# For AI Coding Agents

请先以代码现状为准，再参考后文的研究目标。当前仓库已经落地的是一条可训练的 baseline：

- 训练入口是 `shore_tts/train.py`。
- 模型构建在 `shore_tts/utils/build.py`，当前主干是 `CFM + DiT`，对应 `shore_tts/models/cfm.py` 与 `shore_tts/models/dit.py`。
- 数据管线在 `shore_tts/datasets/dataset.py`，使用 `webdataset` 递归扫描 `data_path` 下的 `*.tar` 分片，在线解码音频与文本，并即时计算 BN-MDCT 特征。
- 特征提取配置在 `shore_tts/configs/mdct.json`，默认是 `sample_rate=44100`、`hop_length=441`、`n_bands=10`，对应特征维度 `451`。
- 默认训练配置在 `shore_tts/configs/pretrain.json`，当前 `data.data_path` 是 `data/parquet`；虽然目录名叫 `parquet`，但代码实际读取的是其中递归存在的 tar shards，而不是 parquet 文件。

因此，后面的“整体流程 / 后续研究”应视为研究说明，其中混有尚未实现或尚未定型的方案。编码时优先核对真实入口、配置和现有模块，不要把 README 里关于未来结构、采样脚本或实验分支的描述误当成已经可用的实现。

# 整体流程

传统的TTS依赖于vocoder, 即使是端到端训练的VITS, 里面也有一个hifigan作为最后的音频生成, 而fish-speech这种则是选择使用audio codec进行编解码, 采用了压缩的思想。

Shore-TTS(暂定命名), 期望实现以下两个突破：

- 不需要vocoder，直接预测一种完全可以用传统算法无损复原的特征。
- 不进行压缩，保留原始信息从而解决音质问题。(可能无法做到)

## Stage 1 : 传统信号处理

首先我们给定一系列的音频 audio -> [wav], 经过BN-MDCT处理(此处等同于以前对音频计算log-mel spectrogram的作用), 得到 D谱 -> [npy][range主要在-3到+3之间], D谱本身可逆, 参考[脚本](./shore_tts/utils/spectrogram.py)。

## Stage 2 : 神经网络骨干

参考 F5-TTS 的做法，我们不显式使用音素和时长预测，这直接解决掉了最烦且老旧的部分，我们选择直接使用 self attention 的方式，用 concat 拼接拼音 token 序列和 D谱 特征，举例如下：

原始音频经过提取后变成 [1,114,51] 大小的特征，注意到时间尺度上长度为 114 帧，那么整个序列的长度都是 114 帧。假设输入文本包含中文，那么会先被转换为类似 F5-TTS 的拼音 token 序列，再与音频特征在时间维上对齐。然后 F5-TTS 采用了一个绝对正弦波位置编码处理文本序列，随后这个张量被送到 ConvNeXt V2 Block 里面，block 输出一个 [114,音频特征维度] 的张量。不过他们的做法是在此之后进行叠加，拼接一个 [114,文本特征维度+音频特征维度] 的张量出来，我们这里可以考虑在序列层次上做叠加。**(后面可能要做实验)**

随后我们要通过一系列神经网络进行处理，最终实现TTS的效果。目前有如下方案：

- 用一种类似vocoder的思路，先使用一个简单的Pre-Net对 D谱 进行降维，然后用DiT预测降维后的信息，之后用一个PostNet复原为 D谱。
- 先使用RealNVP类似的可逆神经网络进行处理, 让 D谱 -> [npy] 变成同等大小的、扩散模型友好的特征 DF(Diffusion Friendly) -> [tensor], 之后采用一个较大的 DiT 网络进行端到端(期望)扩散学习。这样只需要一个 DiT 主干就能生成 DF，过逆 Flow 还原 D谱, 之后可以快速用传统算法反推波形。
- 引入复数域，采用 MCLT (Modulated Complex Lapped Transform) 扩展特征。由于纯 MDCT 极其依赖前后帧的 TDAC (时域混叠消除) 约束，扩散模型微小的预测误差就会打破这种平衡，导致刺耳的伪影（如预回声或金属音）。在此方案中，我们将 MDCT（实部）与 MDST（虚部）结合构成复数域特征。骨干网络（如 DiT 或 Flow Matching）不再痛苦地拟合极其严格的 TDAC 约束，而是去预测具备“平移不变性”的复数频谱（或极坐标下的幅度和相位导数）。这样不仅大大降低了模型的拟合难度，还能在损失函数设计上直接对齐音频的物理规律，最终通过提取实部并执行逆变换，完美重建无伪影的高质量波形。
- 采用解耦级联生成 (Decoupled & Cascaded Generation)。Stage 1 提取的 D谱 实际上已被解耦为结构性极强的“频带包络 (log_mag)”和缺乏平滑结构约束的“归一化残差频谱 (norm_spec)”。包络决定了语音的清晰度、语调和共振峰，极易被网络学习；而残差包含高频细节，直接拟合容易导致高频丢失。在此方案中，我们构建两级生成范式：
    1. **Base Model**: 负责基于文本/音素特征，精准预测低维度的 `log_mag` 能量包络。
    2. **Condition Model**: 以预测出的 `log_mag` 为条件（Condition），指导网络生成高维的精细特征 `norm_spec`。
    更重要的是，得益于 Stage 1 算子的全 PyTorch 可微实现，在训练阶段，我们可以直接让梯度穿过 `BN-MDCT` 逆变换器，在时域上计算 STFT Loss 或引入轻量级鉴别器（Discriminator, 同时可以考虑GAN的对抗蒸馏）

## Stage 3 : 重建音频

由于现在大伙都引入了few-shot开卷，所以我们在可以出声的基础上额外加入一个小目标：引入speaker embedding进行类似cosyvoice的那种few-shot。

不过这一阶段主要的行为依然是把 D谱 -> [npy] 转换为 fake_audio -> [wav]，其中的ODE求解器我们就用Euler即可。

# 后续研究

- 引入解耦生成
- 引入x-prediction
- 优化模型表现力，测试scaling效果


# 注意事项

数据集读取是使用webdataset的，保证一个tar里面全是音频和对应的txt，脚本会自己递归地读子目录等一系列tar。

如果目标是完整复刻 F5-TTS 的模型和训练流程，但把声学特征替换成 MDCT，那么目前还剩下这些关键工作：

- 统一并固定训练配置。当前代码里的主干已经接近 `F5TTS_Base_MDCT`，但仓库默认配置仍然存在不一致之处；例如 `shore_tts/configs/mdct.json` 里还是 `sample_rate=44100`，而 `assets/F5-TTS/src/f5_tts/configs/F5TTS_Base_MDCT.yaml` 使用的是 `24000 / hop_length=100 / n_bands=20`。这部分需要先定成唯一真值。
- 对齐数据契约。现在 `shore_tts/datasets/dataset.py` 走的是 tar 流式在线解码 + 在线 MDCT 提特征；F5-TTS 原版训练则依赖 `duration.json` 和可按帧长排序的数据集接口。若要完整复刻训练行为，需要补齐一套能提供样本时长统计、支持按帧长分桶的数据元信息，或者重写出 tar 版本的等价方案。
- 补上 dynamic batch / frame-based batching。F5-TTS 的关键工程能力之一是按总帧数动态组 batch，而不是固定样本数；当前 `shore_tts/train.py` 仍然是固定 `batch_size`。这一点不补，长语音训练时的吞吐、显存利用率和稳定性都会明显落后于原版。
- 对齐 Trainer 能力。当前仓库已经能训练，但还是“最小闭环”版本；还缺 F5-TTS 原版 Trainer 里的 `grad_accumulation_steps`、更严格的断点续训、按 update 计数的 warmup/decay、以及更完整的多卡训练抽象。
- 规范 checkpoint / resume 语义。现在可以存取 checkpoint，但还没有完全做到和 F5-TTS 一样的“按 update 精确恢复训练进度，包括 dataloader 跳过位置和 batch sampler epoch 状态”。
- 补 inference / finetune / eval 入口。当前仓库主要只有训练入口；如果要说“完整复刻流程”，还需要补 CLI 推理、批量推理、finetune 入口，以及基础评测脚本，至少要能覆盖 F5-TTS 仓库中 `infer/`、`train/finetune_*`、`eval/` 的核心使用路径。
- 明确 MDCT 分支的训练目标边界。当前实现是直接把 F5-TTS 的 `CFM + DiT` 套到 MDCT 特征上，这已经是合理 baseline；但如果后续要进一步追平或超过 mel 方案，还需要继续验证 `log_mag + norm_spec` 的联合建模是否足够，是否要继续做解耦生成、x-prediction、或者时域辅助损失。

简化地说：现在已经有“能训练的 F5-TTS/MDCT baseline”，但距离“完整复刻 F5-TTS 训练体系”还差数据分桶、动态 batch、完整 Trainer、精确续训、以及 inference / finetune / eval 配套入口。
