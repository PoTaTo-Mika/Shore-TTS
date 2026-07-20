# Shore-TTS v2 架构升级方案

## 0. 目标与动机

复现并升级 Shore-TTS，目标是在 8×H20 上训练一个"参数量足够大就不电"的强生成模型（Step 1），
为后续将其蒸馏成 multi-speaker separation 模型（Step 2，对标 htdemucs）打基础。

当前架构（CFM + DiT + ConvNeXt 文本编码 + 拼音）相对于 2025 年的 SOTA 已显老旧，主要短板：

1. **文本注入过浅**：拼音经 ConvNeXtV2 编码后，只在输入处和 `x`/`cond` 拼接一次（`InputEmbedding`），
   后续 22 层 DiT 里文本再不参与任何注意力。文本条件利用严重不足。
2. **CFM 采样需多步**：Euler ODE + EPSS 仍要 16~32 步，NFE 偏高，不利于 Step 2 蒸馏成少步/一步的分离模型。
3. **文本编码器从零训**：没有利用任何预训练语言知识，多音字/混合语言鲁棒性靠 rjieba+pypinyin 硬扛。
4. **参数量偏小**：dim=1152/depth=22 ≈ 500M，未达"不电"的规模门槛。

升级围绕这四点展开。DiT 主干本身保留（scaling 经验最足，HunyuanVideo/Sora/SD3/FLUX 全是 DiT），
改的是文本路径、注意力形式、流匹配目标三处。

---

## 1. 更换 DiT 核心架构为 MMDiT

将 `DiT.transformer_blocks` 从 `DiTBlock`（普通自注意力，文本仅输入期拼接）替换为
`MMDiTBlock`（joint attention，文本与音频 token 每层深度交互）。

### 现状
- `shore_tts/models/diffusion/modules.py` 中 `MMDiTBlock` / `JointAttnProcessor` 已实现，但未接入。
- `shore_tts/models/diffusion/dit.py` 的 `DiT` 实际使用 `AttnProcessor`（单流自注意力）。
- `InputEmbedding` 把 `x(噪声谱) + cond(参考谱) + text_embed` 在通道维拼接后投影，文本没有独立通道。

### 改造
- 主干输入只保留 `x`（噪声 MDCT）与 `cond`（参考 MDCT）的融合；文本 token 作为 **context** 独立流入 MMDiT。
- 每个 `MMDiTBlock` 内部：audio token 与 text token 拼接后做一次 joint self-attention，再分别走各自的 FFN/modulation。
- `context_dim` 对齐到 text encoder 的输出维度（经投影层后）。
- 文本侧的 RoPE 独立计算（文本序列长度与音频不同）。
- 推理时的 CFG：cond/uncond 打包仍按现有方式（`cfg_infer`），uncond 时把 text context 置零、audio cond 置零。

### 收益
- 文本条件每层都参与注意力，对长文本、韵律对齐、多音字消歧都有质变提升。
- 为 Step 2 预留天然条件接口：分离蒸馏时 context 通道可换成 mixture 的 MDCT 特征（或 text + mixture 双条件），
  架构不变，只换条件输入。

---

## 2. Flow 方式从 CFM 改为 Mean Flow

将训练目标从 Conditional Flow Matching（多步 ODE 采样）改为 Mean Flow（一步采样 + CFG 内置）。

### 为什么
- Mean Flow 在 1-NFE（单次前向）下即可生成，CFG 在训练时 baked，推理无需 cond/uncond 双 forward。
- 单步特性天然适合 Step 2 蒸馏：分离任务需要确定性、一步出结果的映射，Mean Flow 的速度场可直接重定向为
  `clean = mixture + u(mixture, 0, 1)` 形式的分离公式。
- 仍属 flow matching 范式，训练稳定性与 CFM 相当，不是跳到 GAN/diffusion 的新坑。

### 改造点
- `shore_tts/models/diffusion/cfm.py` 的 `CFM` 重构（或新建 `MeanFlow` 类）：
  - `forward`：输入 `(x0 噪声, x1 目标, t, cond, text)`，预测速度场的均值参数，按 Mean Flow 论文的目标函数计算损失。
  - `sample`：单步前向出结果（保留可选多步 fallback 用于调试）。
  - CFG 训练时的 drop 策略保留（`audio_drop_prob` / `cond_drop_prob`），但推理不再需要双 forward。
- 频率加权 MSE 损失（`FrequencyWeightedMSELoss`）沿用，作用于速度场预测。

> 注：Mean Flow 的精确损失形式（均值参数化、积分项的处理）需在实现时对照论文逐项核对，
> 此处先定方向，细节在编码阶段落实。

---

## 3. Text Encoder 改为 Qwen3-0.6B 前 16 层（全量 embedding）

把当前的拼音 `TextEmbedding`（`nn.Embedding + ConvNeXtV2Block`）替换为一个用 Qwen3-0.6B 前 16 层初始化的
transformer encoder，端到端微调（**不冻结**，全部 `requires_grad=True`）。
规模约 0.355B（embedding 155M + 16 层 200M），与 DiT 主干 ~0.8B 合计 ~1.15B。

### 设计原则
- 预训练权重仅用于**初始化**，本质是把 Qwen 的语言知识搬进文本编码器，之后随主网络一起训练特化。
- 不做蒸馏、不冻结——8×H20 + Emilia 级数据量足够支撑全网络联训。
- 可给 text encoder 一个比 DiT 主干更小的学习率（如 0.1×），防止早期梯度冲坏预训练特征（通过 param group 实现，
  `Muon_AdamW` 已支持分组）。

### 切片方式（裁层数，全量保留 embedding）
Qwen3-0.6B：28 层 transformer、hidden=1024、GQA、词表 151936。

| 组成 | 全量 | 切片后 |
|---|---|---|
| 输入 embedding | 151936×1024 ≈ 155M | **全量保留** ≈ 155M |
| LM head | ≈155M | **丢弃**（encoder 不需要） |
| transformer 层 | 28 × ~12.5M ≈ 350M | 取前 16 层 ≈ 200M |
| 投影层 1024→text_dim | — | ~1M（从零初始化） |
| **合计** | ~660M | **≈ 355M** |

- **只裁层数**：取前 16 层 transformer，N 由配置文件传入（`text_encoder.num_layers`，默认 16）。
  残差流不断，权重原样加载，预训练知识完整保留。
- **不裁词表**：保留全量 151936 的 embedding。引入 `compact_id ↔ original_id` 重映射表会带来 id 对齐的
  隐患（推理/训练两条路径都要维护映射、易错），全量保留则直接用 Qwen 原始 id，零对齐成本，最稳。
  embedding 的 155M 参数在 8×H20-96G 上可承受，且 DiT 主干做到 ~0.8B 后总规模 ~1.15B 仍合理。
- **额外投影层**：`nn.Linear(1024, text_dim)` 从零初始化，把 Qwen hidden 对齐到 MMDiT 的 `context_dim`。
  这一层很小，端到端训练消化。

### 词表与 token 管线
- 退役拼音管线（`shore_tts/text/tokenizer.py` 的 `PinyinTokenizer`、`checkpoints/vocab.json`）。
- 改用 Qwen3 原始 BPE tokenizer，token 直接来自原始文本（LLM 自己处理多音字/混合语言，Fish-Speech 路线）。
- 不构建子集词表、不做 id 重映射——训练与推理都直接用 Qwen 原始 vocab 和原始 id。

---

## 4. 运行时 tokenizer wrapper `shore_tts/text/qwen_tokenizer.py`

不再构建子集词表、不做 id 重映射，因此**不需要 `build_vocab.py` 这类工具**——直接用 Qwen3 原始 BPE tokenizer 即可。

### 设计
新建 `shore_tts/text/qwen_tokenizer.py`，封装一个轻量 wrapper：
- 加载 Qwen3-0.6B 的原始 tokenizer（`transformers.AutoTokenizer.from_pretrained(qwen_model)`），
  BPE merges 与词表完全不变，使用原始 id。
- `encode(text) → token_ids`：直接调 Qwen tokenizer，返回原始 id 的 tensor/list。
- `batch_encode(texts) → (ids, lengths)`：供 collate 用，按最长补齐（pad 到 batch 内最大长度，pad_id 用 Qwen 的 pad token）。
- 保留一个 `vocab_size` 属性（= 151936），供 `DiT.text_num_embeds` 使用。

### 退役拼音管线
- `shore_tts/text/tokenizer.py` 的 `PinyinTokenizer`、`checkpoints/vocab.json` 不再用于训练/推理主路径。
- `ShoreDataset` 的 collate 侧由拼音 id 改为 Qwen token id（`texts` 字段语义不变，内容换成 Qwen id）。
- 推理路径 `tools/inference.py` 的 `clean_text` + `PinyinTokenizer` 替换为 `QwenTokenizer.encode`。

---

## 5. 模型初始化时加载 Qwen 权重（全量 embedding + 裁层数）

在 `shore_tts/utils/build.py` 的模型构建流程中加入 Qwen 权重加载逻辑。

### 流程
1. 从配置读取 `text_encoder.qwen_model`（HF 路径）与 `text_encoder.num_layers`（N，默认 16）。
2. 加载 Qwen3-0.6B 权重（`transformers.AutoModelForCausalLM` 或直接读 safetensors）。
3. 构建 text encoder 模块（结构与 Qwen3 的前 N 层一致：embedding + N 层 transformer + 各层 RMSNorm）：
   - **输入 embedding**：`nn.Embedding(151936, 1024)`，权重 = Qwen 的 `model.embed_tokens.weight`（**全量原样**，不做任何行选择）。
   - **前 N 层 transformer**：`qwen.layers[:N]`，逐层复制权重，结构完全对齐 Qwen3（GQA、SwiGLU、RoPE、RMSNorm）。
   - **丢弃** `qwen.lm_head` 与 `qwen.norm`（最末层 final norm，encoder 不需要；各层内 RMSNorm 保留）。
   - **投影层** `proj = nn.Linear(1024, text_dim)`，从零初始化，把 Qwen hidden 对齐到 MMDiT 的 `context_dim`。
4. 将该 encoder 作为 MMDiT 的文本 context 来源，替换原 `TextEmbedding`。
5. 整个 text encoder 全部 `requires_grad=True`，端到端微调。

### 配置参数（新增到 `pretrain.json`，新开 `model.text_encoder` 块）
```json
"text_encoder": {
  "type": "qwen3",
  "qwen_model": "Qwen/Qwen3-0.6B",
  "hidden_dim": 1024,
  "num_layers": 16,
  "causal": true,
  "lr_scale": 0.1
}
```
- `causal: true`：保持 Qwen 的 causal attention（扰动最小，预训练匹配度最高）；效果不够再试 bidirectional。
- `lr_scale`：text encoder 相对主学习率的缩放，传给优化器的 param group（`Muon_AdamW` 分组支持）。
- 不再有 `vocab_file` 字段——直接用 Qwen 原始 vocab，无需子集词表文件。

---

## 6. 端到端训练与推理

### 训练
- 整个模型（Qwen encoder + 投影层 + MMDiT 主干 + MDCT spec 模块）端到端联训，全部可训练。
- 优化器：`Muon_AdamW` 分组——text encoder 参数走 `lr * lr_scale`，其余走 `lr`。`build_optimizer` 增加 param group 支持。
- 损失：Mean Flow 目标 + 频率加权 MSE，作用于掩码区域（span masking 保留）。
- 数据：`ShoreDataset` 不变，只是 `texts` 字段从拼音 id 改为 Qwen 原始 id（collate 侧由新 tokenizer 产生）。
- EMA、检查点、resume 逻辑沿用 `Trainer`，checkpoint 里增加 text encoder 权重（自然包含在 `model_state_dict` 中）。

### 推理
- `tools/inference.py` 的 `load_model` 增加加载 Qwen tokenizer。
- 文本输入直接走 Qwen tokenizer wrapper，不再走 `clean_text` + `PinyinTokenizer`。
- Mean Flow 单步采样，CFG 已 baked，去掉 cond/uncond 双 forward。
- 其余（参考音频、时长控制、MDCT 逆变换还原波形）不变。

### 检查点格式
沿用 `config.json + vocab.json + model.pth`。`vocab.json` 不再是子集词表——text encoder 用 Qwen 原始 vocab，
词表由 HF 模型自带，checkpoint 只需保存 `config.json`（含 `text_encoder.qwen_model` 路径）即可在推理时重建 tokenizer。
`model.pth` 里 `text_encoder.*` 权重（全量 embedding + 16 层 transformer + 投影层）随主模型一起保存/加载。

---

## 7. 文件改动地图

| 文件 | 改动 |
|---|---|
| `shore_tts/models/diffusion/dit.py` | `DiT` 改用 `MMDiTBlock`；移除 `InputEmbedding` 中的文本拼接，文本走 context；新增 text encoder 接入 |
| `shore_tts/models/diffusion/modules.py` | `MMDiTBlock`/`JointAttnProcessor` 接入主路径（已实现，需联调）；`TextEmbedding` 退役或保留为 fallback |
| `shore_tts/models/diffusion/cfm.py` | 重构为 Mean Flow（`forward`/`sample`），或新建 `mean_flow.py` |
| `shore_tts/utils/build.py` | 新增 Qwen 权重加载（全量 embedding + 裁层数）+ text encoder 构建；优化器 param group 支持 `lr_scale` |
| `shore_tts/utils/trainer.py` | 数据 collate 适配新 tokenizer id；其余（EMA/ckpt/resume）基本不变 |
| `shore_tts/text/qwen_tokenizer.py` | **新建**：Qwen 原始 tokenizer wrapper（不做词表裁切/重映射） |
| `shore_tts/configs/pretrain.json` | 新增 `model.text_encoder` 配置块；MDCT/优化器/调度参数按规模调整 |
| `tools/inference.py` | `load_model` 加载 Qwen tokenizer；文本路径改走新 tokenizer；Mean Flow 单步采样 |
| `shore_tts/text/tokenizer.py` | 保留代码但不再用于训练/推理主路径（拼音管线退役） |

> 不再需要 `tools/building/build_vocab.py`——保留全量 Qwen vocab，不做子集词表构建。

---

## 8. 待确认与风险

1. **Mean Flow 损失的精确形式**需对照论文实现，是本方案最大的不确定性。
2. **Qwen3-0.6B 权重可访问性**：需确认 HF 能拉到（公司网络/镜像）。
3. **层数**：定 **16 层**（text encoder ≈ 0.355B），DiT 主干目标 ~0.8B，合计 ~1.15B。
4. **causal vs bidirectional**：先 causal，预留切换。
5. **MDCT hop**：100（120 维、240 帧/s）还是 200（220 维、120 帧/s）？MMDiT 下文本+音频 token 都进注意力，序列更长，倾向 100。
6. **DiT 主干规模**：目标 ~0.8B，具体 dim/depth 组合待定（如 dim=1536/depth=24 ≈ 0.8B），影响 batch size 与显存。
