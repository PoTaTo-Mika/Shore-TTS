import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """
    正弦余弦位置编码，适用于TTS系统的文本编码器
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # 创建位置编码矩阵 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 偶数位置使用sin，奇数位置使用cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # [max_len, d_model] -> [1, max_len, d_model] 适用于 [batch, seq, dim]
        pe = pe.unsqueeze(0)
        # 注册为buffer，不参与梯度更新
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: 输入张量，形状为 [batch_size, seq_len, d_model] 
        # 或 [seq_len, batch_size, d_model]
        if x.dim() == 3:
            if x.size(1) <= self.pe.size(1):  # [batch, seq, dim] 格式
                return x + self.pe[:, :x.size(1), :]
            else:
                # 如果序列长度超过预设的max_len，需要扩展位置编码
                raise ValueError(f"序列长度 {x.size(1)} 超过了位置编码的最大长度 {self.pe.size(1)}")
        else:
            raise ValueError(f"输入张量的维度应该是3，但得到了 {x.dim()}")

class TextEncoder(nn.Module):
    def __init__(self,
                 vocab_size = 221 + 1, # 221个音素 + 1个padding
                 embedding_dim = 128, # 必须要和d_model一致
                 hidden_dim = 128,
                 output_dim = 128,
                 n_layers = 4,
                 n_heads = 8,
                 d_model = 128, # model dim
                 dropout = 0.1,
                 max_len = 1024
    ):
        super(TextEncoder, self).__init__()
        # 输入的batch格式：
        # (padded_mels, padded_phoneme_ids, mel_lengths, phoneme_lengths)
        # 形状分别为：
        # padded_mels:        [batch_size, n_mels, max_mel_length]  
        # padded_phoneme_ids: [batch_size, max_phoneme_length]        
        # mel_lengths:        [batch_size]                             
        # phoneme_lengths:    [batch_size]       
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.output_dim = output_dim
        # 输入一个[batch_size, max_phoneme_length]的音素序列 -> [batch_size, max_phoneme_length, embedding_dim]
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        # 如果embedding_dim != d_model，直接报错
        if embedding_dim != d_model:
            raise ValueError("embedding_dim must be equal to d_model")
          
        # Transformer编码器
        d_ff = hidden_dim * 4  # 前馈网络的隐藏层维度，通常是d_model的4倍
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        # 输出投影层，将d_model维度映射到output_dim
        self.output_projection = nn.Linear(d_model, output_dim)
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
    def create_padding_mask(self, phoneme_ids, phoneme_lengths):
        """
        创建padding mask，用于屏蔽padding位置
        Args:
            phoneme_ids: [batch_size, max_phoneme_length] 音素ID序列
            phoneme_lengths: [batch_size] 每个序列的实际长度
        Returns:
            mask: [batch_size, max_phoneme_length] True表示padding位置
        """
        batch_size, max_len = phoneme_ids.shape
        # 创建位置索引 [batch_size, max_len]
        positions = torch.arange(max_len, device=phoneme_ids.device).unsqueeze(0).expand(batch_size, -1)
        # 创建mask，True表示padding位置
        mask = positions >= phoneme_lengths.unsqueeze(1)
        return mask
        
    def forward(self, phoneme_ids, phoneme_lengths):
        # 1. 词嵌入 [batch_size, max_phoneme_length] -> [batch_size, max_phoneme_length, d_model]
        embedded = self.embedding(phoneme_ids)
        # 2. 缩放嵌入（Transformer标准做法）
        embedded = embedded * math.sqrt(self.d_model)
        # 3. 添加位置编码
        embedded = self.pos_encoding(embedded)
        # 4. Dropout
        embedded = self.dropout(embedded)
        # 5. 创建padding mask
        padding_mask = self.create_padding_mask(phoneme_ids, phoneme_lengths)
        # 6. Transformer编码器需要的输入格式是 [seq_len, batch_size, d_model]
        # 所以需要转置
        embedded = embedded.transpose(0, 1)  # [max_phoneme_length, batch_size, d_model]
        # 7. 通过Transformer编码器
        # src_key_padding_mask: [batch_size, seq_len] True表示padding位置
        encoded = self.transformer_encoder(embedded, src_key_padding_mask=padding_mask)
        # 8. 转回 [batch_size, max_phoneme_length, d_model]
        encoded = encoded.transpose(0, 1)
        # 9. 输出投影
        encoded = self.output_projection(encoded)
        
        # encoded -> [batch_size, max_phoneme_length, output_dim]
        # padding_mask -> [batch_size, max_phoneme_length]

        return encoded, padding_mask

# 往下的没有经过测试

class MultiHeadCrossAttention(nn.Module):
    """
    多头交叉注意力机制，用于文本和音频的对齐
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadCrossAttention, self).__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Query, Key, Value投影层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, query, key, value, key_padding_mask=None):
        """
        Args:
            query: [batch_size, tgt_len, d_model] (解码器状态)
            key: [batch_size, src_len, d_model] (编码器输出)
            value: [batch_size, src_len, d_model] (编码器输出)
            key_padding_mask: [batch_size, src_len] (True表示padding)
        Returns:
            output: [batch_size, tgt_len, d_model]
            attention_weights: [batch_size, n_heads, tgt_len, src_len]
        """
        batch_size, tgt_len, _ = query.shape
        src_len = key.shape[1]
        
        # 线性变换并重塑为多头格式
        Q = self.w_q(query).view(batch_size, tgt_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, src_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, src_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # 应用padding mask
        if key_padding_mask is not None:
            # key_padding_mask: [batch_size, src_len] -> [batch_size, 1, 1, src_len]
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores.masked_fill_(mask, -1e9)
        
        # Softmax得到注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力权重
        context = torch.matmul(attention_weights, V)
        
        # 重塑并通过输出投影
        context = context.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.d_model)
        output = self.w_o(context)
        
        return output, attention_weights


class TransformerDecoderLayer(nn.Module):
    """
    Transformer解码器层，包含自注意力、交叉注意力和前馈网络
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        
        # 自注意力
        self.self_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        
        # 交叉注意力
        self.cross_attention = MultiHeadCrossAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(d_model)
        
    def forward(self, tgt, memory, tgt_mask=None, memory_key_padding_mask=None):
        """
        Args:
            tgt: [batch_size, tgt_len, d_model] 目标序列（解码器输入）
            memory: [batch_size, src_len, d_model] 编码器输出
            tgt_mask: [tgt_len, tgt_len] 因果mask
            memory_key_padding_mask: [batch_size, src_len] 编码器padding mask
        """
        # 1. 自注意力
        tgt2, _ = self.self_attention(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = self.norm1(tgt + tgt2)
        
        # 2. 交叉注意力
        tgt2, cross_attn_weights = self.cross_attention(tgt, memory, memory, memory_key_padding_mask)
        tgt = self.norm2(tgt + tgt2)
        
        # 3. 前馈网络
        tgt2 = self.feed_forward(tgt)
        tgt = self.norm3(tgt + tgt2)
        
        return tgt, cross_attn_weights


class MelDecoder(nn.Module):
    """
    基于Transformer的Mel频谱图解码器
    """
    def __init__(self, 
                 n_mels=128,
                 d_model=128,
                 n_heads=8,
                 n_layers=6,
                 d_ff=512,
                 dropout=0.1,
                 max_len=2000):
        super(MelDecoder, self).__init__()
        
        self.n_mels = n_mels
        self.d_model = d_model
        self.max_len = max_len
        
        # Mel频谱图的输入投影
        self.mel_projection = nn.Linear(n_mels, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Transformer解码器层
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # 输出投影层
        self.mel_output = nn.Linear(d_model, n_mels)
        self.stop_output = nn.Linear(d_model, 1)  # 停止标记预测
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def create_causal_mask(self, seq_len, device):
        """创建因果mask，防止解码器看到未来的信息"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
        
    def forward(self, encoder_output, encoder_padding_mask, mel_target=None, max_len=None):
        """
        Args:
            encoder_output: [batch_size, src_len, d_model] 编码器输出
            encoder_padding_mask: [batch_size, src_len] 编码器padding mask
            mel_target: [batch_size, tgt_len, n_mels] 目标mel频谱图（训练时使用）
            max_len: int 推理时的最大长度
        Returns:
            mel_outputs: [batch_size, tgt_len, n_mels]
            stop_outputs: [batch_size, tgt_len, 1]
            attention_weights: List of attention weights from each layer
        """
        batch_size = encoder_output.shape[0]
        device = encoder_output.device
        
        # 修复逻辑：如果提供了mel_target，就使用teacher forcing，无论训练还是验证模式
        if mel_target is not None:
            # Teacher forcing模式：使用目标mel序列
            tgt_len = mel_target.shape[1]
            
            # 将mel频谱图投影到d_model维度
            decoder_input = self.mel_projection(mel_target)
            decoder_input = decoder_input * math.sqrt(self.d_model)
            decoder_input = self.pos_encoding(decoder_input)
            decoder_input = self.dropout(decoder_input)
            
            # 创建因果mask
            causal_mask = self.create_causal_mask(tgt_len, device)
            
            # 通过解码器层
            all_attention_weights = []
            for layer in self.decoder_layers:
                decoder_input, attn_weights = layer(
                    decoder_input, encoder_output, 
                    tgt_mask=causal_mask,
                    memory_key_padding_mask=encoder_padding_mask
                )
                all_attention_weights.append(attn_weights)
            
            # 输出投影
            mel_outputs = self.mel_output(decoder_input)
            stop_outputs = torch.sigmoid(self.stop_output(decoder_input))
            
            return mel_outputs, stop_outputs, all_attention_weights
            
        else:
            # 推理模式：自回归生成（只有在没有提供mel_target时才使用）
            if max_len is None:
                max_len = self.max_len
                
            # 初始化输出
            mel_outputs = []
            stop_outputs = []
            all_attention_weights = []
            
            # 初始输入（零向量）
            decoder_input = torch.zeros(batch_size, 1, self.n_mels, device=device)
            
            for step in range(max_len):
                # 投影当前输入
                current_input = self.mel_projection(decoder_input)
                current_input = current_input * math.sqrt(self.d_model)
                current_input = self.pos_encoding(current_input)
                current_input = self.dropout(current_input)
                
                # 创建因果mask
                tgt_len = current_input.shape[1]
                causal_mask = self.create_causal_mask(tgt_len, device)
                
                # 通过解码器层
                step_attention_weights = []
                for layer in self.decoder_layers:
                    current_input, attn_weights = layer(
                        current_input, encoder_output,
                        tgt_mask=causal_mask,
                        memory_key_padding_mask=encoder_padding_mask
                    )
                    step_attention_weights.append(attn_weights)
                
                # 预测当前步的输出
                mel_output = self.mel_output(current_input[:, -1:, :])  # 只取最后一步
                stop_output = torch.sigmoid(self.stop_output(current_input[:, -1:, :]))
                
                mel_outputs.append(mel_output)
                stop_outputs.append(stop_output)
                all_attention_weights.append(step_attention_weights)
                
                # 检查是否应该停止
                if torch.all(stop_output > 0.5):
                    break
                    
                # 更新解码器输入
                decoder_input = torch.cat([decoder_input, mel_output], dim=1)
            
            # 拼接所有输出
            mel_outputs = torch.cat(mel_outputs, dim=1)
            stop_outputs = torch.cat(stop_outputs, dim=1)
            
            return mel_outputs, stop_outputs, all_attention_weights


class PostNet(nn.Module):
    """
    后处理网络，用于改善生成的mel频谱图质量
    """
    def __init__(self, n_mels=128, n_layers=5, kernel_size=5, n_channels=512, dropout=0.1):
        super(PostNet, self).__init__()
        
        self.convolutions = nn.ModuleList()
        
        # 第一层
        self.convolutions.append(
            nn.Sequential(
                nn.Conv1d(n_mels, n_channels, kernel_size, padding=(kernel_size-1)//2),
                nn.BatchNorm1d(n_channels),
                nn.Tanh(),
                nn.Dropout(dropout)
            )
        )
        
        # 中间层
        for _ in range(n_layers - 2):
            self.convolutions.append(
                nn.Sequential(
                    nn.Conv1d(n_channels, n_channels, kernel_size, padding=(kernel_size-1)//2),
                    nn.BatchNorm1d(n_channels),
                    nn.Tanh(),
                    nn.Dropout(dropout)
                )
            )
        
        # 最后一层
        self.convolutions.append(
            nn.Sequential(
                nn.Conv1d(n_channels, n_mels, kernel_size, padding=(kernel_size-1)//2),
                nn.BatchNorm1d(n_mels),
                nn.Dropout(dropout)
            )
        )
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, n_mels]
        Returns:
            residual: [batch_size, seq_len, n_mels]
        """
        # 转换为Conv1d格式 [batch_size, n_mels, seq_len]
        x = x.transpose(1, 2)
        
        for conv in self.convolutions:
            x = conv(x)
        
        # 转换回原格式 [batch_size, seq_len, n_mels]
        x = x.transpose(1, 2)
        
        return x


class ShoreTTS(nn.Module):
    """
    完整的Shore TTS模型，基于Transformer架构
    """
    def __init__(self,
                 # TextEncoder参数
                 vocab_size=223,
                 text_embedding_dim=128,
                 text_hidden_dim=128,
                 text_n_layers=4,
                 text_n_heads=8,
                 text_d_model=128,
                 # MelDecoder参数
                 n_mels=128,
                 decoder_d_model=128,
                 decoder_n_heads=8,
                 decoder_n_layers=6,
                 decoder_d_ff=512,
                 # PostNet参数
                 postnet_n_layers=5,
                 postnet_kernel_size=5,
                 postnet_n_channels=512,
                 # 通用参数
                 dropout=0.1,
                 max_text_len=1024,
                 max_mel_len=2000):
        super(ShoreTTS, self).__init__()
        
        # 文本编码器
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            embedding_dim=text_embedding_dim,
            hidden_dim=text_hidden_dim,
            output_dim=decoder_d_model,  # 确保输出维度匹配解码器
            n_layers=text_n_layers,
            n_heads=text_n_heads,
            d_model=text_d_model,
            dropout=dropout,
            max_len=max_text_len
        )
        
        # Mel解码器
        self.mel_decoder = MelDecoder(
            n_mels=n_mels,
            d_model=decoder_d_model,
            n_heads=decoder_n_heads,
            n_layers=decoder_n_layers,
            d_ff=decoder_d_ff,
            dropout=dropout,
            max_len=max_mel_len
        )
        
        # 后处理网络
        self.postnet = PostNet(
            n_mels=n_mels,
            n_layers=postnet_n_layers,
            kernel_size=postnet_kernel_size,
            n_channels=postnet_n_channels,
            dropout=dropout
        )
        
    def forward(self, phoneme_ids, phoneme_lengths, mel_target=None, max_mel_len=None):
        """
        Args:
            phoneme_ids: [batch_size, max_phoneme_length] 音素ID序列
            phoneme_lengths: [batch_size] 音素序列长度
            mel_target: [batch_size, max_mel_length, n_mels] 目标mel频谱图（训练时）
            max_mel_len: int 推理时的最大mel长度
        Returns:
            mel_outputs: [batch_size, mel_length, n_mels] 原始mel输出
            mel_outputs_postnet: [batch_size, mel_length, n_mels] 后处理mel输出
            stop_outputs: [batch_size, mel_length, 1] 停止标记预测
            attention_weights: 注意力权重
        """
        # 1. 文本编码
        encoder_output, encoder_padding_mask = self.text_encoder(phoneme_ids, phoneme_lengths)
        
        # 2. Mel解码
        mel_outputs, stop_outputs, attention_weights = self.mel_decoder(
            encoder_output, encoder_padding_mask, mel_target, max_mel_len
        )
        
        # 3. 后处理
        mel_residual = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_residual
        
        return mel_outputs, mel_outputs_postnet, stop_outputs, attention_weights
    
    def inference(self, phoneme_ids, phoneme_lengths, max_mel_len=1000):
        """推理接口"""
        self.eval()
        with torch.no_grad():
            return self.forward(phoneme_ids, phoneme_lengths, max_mel_len=max_mel_len)


if __name__ == "__main__":
    import sys
    import os
    
    # 添加项目根目录到路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    sys.path.insert(0, root_dir)
    
    from shore_tts.datasets.shore_datasets import ShoreDataset
    from torch.utils.data import DataLoader
    
    print("=" * 60)
    print("=== Shore TTS 模型测试 ===")
    print("=" * 60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    try:
        # 1. 创建数据集和数据加载器
        print("\n1. 创建数据集...")
        dataset = ShoreDataset(
            mel_list_path="data/mel_list.list",
            pinyin_list_path="data/pinyin_list.list",
            device=device,
            max_mel_length=3000,
            min_mel_length=100
        )
        
        print(f"数据集大小: {len(dataset)}")
        
        # 创建DataLoader
        dataloader = DataLoader(
            dataset, 
            batch_size=2, 
            shuffle=False, 
            collate_fn=dataset.collate_fn
        )
        
        print("数据加载器创建成功!")
        
        # 2. 创建模型
        print("\n2. 创建Shore TTS模型...")
        model = ShoreTTS(
            # TextEncoder参数
            vocab_size=222,
            text_embedding_dim=128,
            text_hidden_dim=128,
            text_n_layers=4,
            text_n_heads=8,
            text_d_model=128,
            # MelDecoder参数
            n_mels=128,  # 与你的数据集设置一致
            decoder_d_model=128,
            decoder_n_heads=8,
            decoder_n_layers=6,
            decoder_d_ff=512,
            # PostNet参数
            postnet_n_layers=5,
            postnet_kernel_size=5,
            postnet_n_channels=512,
            # 通用参数
            dropout=0.1,
            max_text_len=1024,
            max_mel_len=2000
        ).to(device)
        
        print("模型创建成功!")
        
        # 计算模型参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"总参数数量: {total_params:,}")
        print(f"可训练参数数量: {trainable_params:,}")
        
        # 3. 获取一个batch的数据
        print("\n3. 获取测试数据...")
        batch_data = next(iter(dataloader))
        padded_mels, padded_phoneme_ids, mel_lengths, phoneme_lengths = batch_data
        
        print(f"Batch数据形状:")
        print(f"  padded_mels: {padded_mels.shape}")
        print(f"  padded_phoneme_ids: {padded_phoneme_ids.shape}")
        print(f"  mel_lengths: {mel_lengths.tolist()}")
        print(f"  phoneme_lengths: {phoneme_lengths.tolist()}")
        
        # 4. 测试模型前向传播
        print("\n4. 测试模型前向传播...")
        model.train()  # 设置为训练模式
        
        # 准备输入数据
        # 注意：数据集返回的mel格式是 [batch_size, n_mels, max_mel_length]
        # 但模型期望的格式是 [batch_size, max_mel_length, n_mels]
        mel_target = padded_mels.transpose(1, 2)  # [batch_size, max_mel_length, n_mels]
        
        print(f"转换后的mel_target形状: {mel_target.shape}")
        
        # 前向传播
        with torch.no_grad():  # 先不计算梯度，只测试前向传播
            mel_outputs, mel_outputs_postnet, stop_outputs, attention_weights = model(
                phoneme_ids=padded_phoneme_ids,
                phoneme_lengths=phoneme_lengths,
                mel_target=mel_target
            )
        
        print("前向传播成功!")
        print(f"输出形状:")
        print(f"  mel_outputs: {mel_outputs.shape}")
        print(f"  mel_outputs_postnet: {mel_outputs_postnet.shape}")
        print(f"  stop_outputs: {stop_outputs.shape}")
        print(f"  attention_weights层数: {len(attention_weights)}")
        if attention_weights:
            print(f"  第一层attention权重形状: {attention_weights[0].shape}")
        
        # 5. 测试损失计算和梯度传播
        print("\n5. 测试损失计算和梯度传播...")
        
        # 定义损失函数
        mel_criterion = nn.MSELoss()
        stop_criterion = nn.BCELoss()
        
        # 创建优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播（这次计算梯度）
        mel_outputs, mel_outputs_postnet, stop_outputs, attention_weights = model(
            phoneme_ids=padded_phoneme_ids,
            phoneme_lengths=phoneme_lengths,
            mel_target=mel_target
        )
        
        # 计算损失
        # Mel损失：只计算有效长度内的损失
        mel_loss = 0
        postnet_loss = 0
        stop_loss = 0
        
        for i in range(mel_target.shape[0]):  # 遍历batch中的每个样本
            actual_mel_length = mel_lengths[i].item()
            
            # 只计算有效长度内的mel损失
            mel_loss += mel_criterion(
                mel_outputs[i, :actual_mel_length, :],
                mel_target[i, :actual_mel_length, :]
            )
            postnet_loss += mel_criterion(
                mel_outputs_postnet[i, :actual_mel_length, :],
                mel_target[i, :actual_mel_length, :]
            )
            
            # 创建stop target（最后一帧为1，其他为0）
            stop_target = torch.zeros(actual_mel_length, 1, device=device)
            stop_target[-1, 0] = 1.0  # 最后一帧设为1
            
            stop_loss += stop_criterion(
                stop_outputs[i, :actual_mel_length, :],
                stop_target
            )
        
        # 平均损失
        mel_loss = mel_loss / mel_target.shape[0]
        postnet_loss = postnet_loss / mel_target.shape[0]
        stop_loss = stop_loss / mel_target.shape[0]
        
        # 总损失
        total_loss = mel_loss + postnet_loss + stop_loss
        
        print(f"损失值:")
        print(f"  mel_loss: {mel_loss.item():.6f}")
        print(f"  postnet_loss: {postnet_loss.item():.6f}")
        print(f"  stop_loss: {stop_loss.item():.6f}")
        print(f"  total_loss: {total_loss.item():.6f}")
        
        # 反向传播
        total_loss.backward()
        
        # 检查梯度
        grad_norm = 0
        param_count = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_grad_norm = param.grad.data.norm(2)
                grad_norm += param_grad_norm.item() ** 2
                param_count += 1
        
        grad_norm = grad_norm ** (1. / 2)
        
        print(f"梯度计算成功!")
        print(f"  有梯度的参数数量: {param_count}")
        print(f"  梯度范数: {grad_norm:.6f}")
        
        # 更新参数
        optimizer.step()
        print("参数更新成功!")
        
        # 6. 测试推理模式
        print("\n6. 测试推理模式...")
        model.eval()
        
        with torch.no_grad():
            # 只使用phoneme输入进行推理
            mel_outputs_inf, mel_outputs_postnet_inf, stop_outputs_inf, attention_weights_inf = model.inference(
                phoneme_ids=padded_phoneme_ids,
                phoneme_lengths=phoneme_lengths,
                max_mel_len=500  # 限制推理长度
            )
        
        print("推理模式测试成功!")
        print(f"推理输出形状:")
        print(f"  mel_outputs: {mel_outputs_inf.shape}")
        print(f"  mel_outputs_postnet: {mel_outputs_inf.shape}")
        print(f"  stop_outputs: {stop_outputs_inf.shape}")
        
        # 7. 测试注意力权重可视化信息
        print("\n7. 注意力权重分析...")
        if attention_weights:
            first_layer_attn = attention_weights[0]  # [batch_size, n_heads, tgt_len, src_len]
            print(f"第一层注意力权重形状: {first_layer_attn.shape}")
            
            # 分析第一个样本的注意力权重
            sample_attn = first_layer_attn[0]  # [n_heads, tgt_len, src_len]
            print(f"第一个样本的注意力权重:")
            print(f"  头数: {sample_attn.shape[0]}")
            print(f"  目标长度: {sample_attn.shape[1]}")
            print(f"  源长度: {sample_attn.shape[2]}")
            
            # 计算注意力权重的统计信息
            attn_mean = sample_attn.mean().item()
            attn_std = sample_attn.std().item()
            attn_max = sample_attn.max().item()
            attn_min = sample_attn.min().item()
            
            print(f"  注意力权重统计:")
            print(f"    均值: {attn_mean:.6f}")
            print(f"    标准差: {attn_std:.6f}")
            print(f"    最大值: {attn_max:.6f}")
            print(f"    最小值: {attn_min:.6f}")
        
        print("\n" + "=" * 60)
        print("=== 所有测试通过! 模型可以用于训练 ===")
        print("=" * 60)
        
        # 8. 总结测试结果
        print(f"\n测试总结:")
        print(f"✅ 数据集加载: 成功")
        print(f"✅ 模型创建: 成功 ({trainable_params:,} 个可训练参数)")
        print(f"✅ 前向传播: 成功")
        print(f"✅ 损失计算: 成功 (总损失: {total_loss.item():.6f})")
        print(f"✅ 梯度传播: 成功 (梯度范数: {grad_norm:.6f})")
        print(f"✅ 参数更新: 成功")
        print(f"✅ 推理模式: 成功")
        print(f"✅ 注意力机制: 成功")
        
        print(f"\n模型已准备好进行训练!")
        
    except FileNotFoundError as e:
        print(f"❌ 文件未找到: {e}")
        print("请确保 data/mel_list.list 和 data/pinyin_list.list 文件存在")
        
    except Exception as e:
        print(f"❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        
        # 提供调试信息
        print(f"\n调试信息:")
        print(f"当前工作目录: {os.getcwd()}")
        print(f"Python路径: {sys.path[:3]}...")  # 只显示前3个路径

