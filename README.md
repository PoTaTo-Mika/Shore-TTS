# Shore-TTS: Vocoder is not all you need

This is still an early demo of our work, we will push the model better and release relative technical report in the future.

All rights reserved to [Fish Audio](https://fish.audio/).

# Quick Start 

## For AI Agents

Please refer to [AGENT.md](./assets/AGENTS.md) to get a brief introduction of this repository.

## For Human

Please refer to these documents:

- [Inference](./assets/docs/inference.md)
- [Pre-training](./assets/docs/training.md)
- [Pending] [Finetune](./assets/docs/finetune.md) 
- [Pending] [Examples](./assets/docs/examples.md)

## Credits

We borrow a lot of code from [F5-TTS](https://github.com/SWivid/F5-TTS).

## Experiments

### AE 不够原教旨

1. 加AE部分，变成latent方案，但是可以做到更换特征，消灭显式vocoder
2. 高低频分离生成，每一支只负责一半的频域，从而把压力从高dim压下去，也可以优化频域损失函数

### GAN对抗训练

1. 从一开始训练就引入判别器，只是把生成器变成cfm的模型
2. 先预训练出基本可用权重，随后引入判别器进行高频音质优化
3. 消融，比对MPD,MRD,FWD,FSD等多种组合的效果
4. 步数蒸馏

### 换flow

1. Reflow替换CFM
2. MeanFlow替换CFM

### 文本端

1. 引入ByteTokenizer，彻底杀死g2p

### 实验性内容

1. control-net做emotion/instruct编辑
2. 基于预训练权重的知识蒸馏成一个分离模型