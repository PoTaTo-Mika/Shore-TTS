import torch
import torch.nn as nn
import torch.nn.functional as F

####### Tacotron 1的CBHG模型实现 ########

class Conv1DBank(nn.Module): # 1D卷积组
    def __init__(self, K, input_dim, conv_channels): 
        """
        Args:
            K (int): 卷积核数量
            input_dim (int): 输入维度
            conv_channels (int): 卷积层输出通道数
        """
        # 因为我们不需要postnet了，所以input_dim就是词嵌入维度
        # K对应卷积核数量
        # input_dim = 词嵌入dim
        super(Conv1DBank, self).__init__()
        self.K = K
        self.input_dim = input_dim
        self.conv_channels = conv_channels
        
        # 创建K个不同宽度的1D卷积层
        # 第k个卷积层的核宽度为k (k=1,2,...,K)
        # 使用 padding=(k-1)//2 确保输出长度与输入长度相同
        self.convs = nn.ModuleList([
            nn.Conv1d(input_dim, conv_channels, kernel_size=k, 
                     stride=1, padding=(k-1)//2, bias=False)
            for k in range(1, K+1)
        ]) # 论文里是16个
        
        # 为每个卷积层添加批量归一化
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(conv_channels) for _ in range(K)
        ])
        
        # 最大池化层
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
        
    def forward(self, x):
        """
        Args:
            x (Tensor): 输入张量，形状为 (batch_size, input_dim, seq_len)
        Returns:
            Tensor: 输出张量，形状为 (batch_size, K*conv_channels, seq_len)
        """
        conv_outputs = []
        target_length = x.size(2)  # 目标序列长度
        
        # 对每个卷积层进行前向传播
        for conv, bn in zip(self.convs, self.batch_norms):
            # 卷积 -> 批量归一化 -> ReLU激活
            conv_out = conv(x)
            conv_out = bn(conv_out)
            conv_out = F.relu(conv_out)
            
            # 确保输出长度与输入长度一致
            if conv_out.size(2) > target_length:
                conv_out = conv_out[:, :, :target_length]
            elif conv_out.size(2) < target_length:
                # 如果输出长度小于目标长度，进行填充
                padding_needed = target_length - conv_out.size(2)
                conv_out = F.pad(conv_out, (0, padding_needed), mode='replicate')
            
            conv_outputs.append(conv_out)
        
        # 将所有卷积输出沿通道维度拼接
        # 形状: (batch_size, K*conv_channels, seq_len)
        concat_output = torch.cat(conv_outputs, dim=1)
        
        # 应用最大池化以增加局部不变性
        pooled_output = self.max_pool(concat_output)
        
        # 调整长度以匹配原始序列长度
        if pooled_output.size(2) > target_length:
            pooled_output = pooled_output[:, :, :target_length]
        elif pooled_output.size(2) < target_length:
            padding_needed = target_length - pooled_output.size(2)
            pooled_output = F.pad(pooled_output, (0, padding_needed), mode='replicate')
        
        return pooled_output

class HighWayNet(nn.Module): # 高速公路网络
    def __init__(self, input_dim, num_layers=4):
        """
        Args:
            input_dim (int): 输入特征维度
            num_layers (int): 高速公路层数，默认为4层
        """
        super(HighWayNet, self).__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        
        # 创建多层高速公路网络
        # 根据论文，Highway网络使用全连接层
        self.highway_layers = nn.ModuleList()
        
        for i in range(num_layers):
            # 每一层包含两个全连接层：H(x)和T(x)
            # H(x): 非线性变换层
            h_layer = nn.Linear(input_dim, input_dim)
            # T(x): 变换门控层 (Transform Gate)
            t_layer = nn.Linear(input_dim, input_dim)
            
            self.highway_layers.append(nn.ModuleDict({
                'H': h_layer,  # 非线性变换
                'T': t_layer   # 变换门控
            }))
    
    def forward(self, x):
        """
        Args:
            x (Tensor): 输入张量，形状为 (batch_size, input_dim, seq_len)
        Returns:
            Tensor: 输出张量，形状与输入相同 (batch_size, input_dim, seq_len)
        """
        # 高速公路网络的核心思想：
        # y = H(x) * T(x) + x * (1 - T(x))
        # 其中 H(x) 是非线性变换，T(x) 是变换门控
        
        # 转换维度以适应Linear层: (batch_size, input_dim, seq_len) -> (batch_size, seq_len, input_dim)
        x = x.transpose(1, 2)
        
        for layer in self.highway_layers:
            # 计算非线性变换 H(x)
            H = F.relu(layer['H'](x))
            
            # 计算变换门控 T(x)，使用sigmoid激活确保输出在[0,1]范围内
            T = torch.sigmoid(layer['T'](x))
            
            # 应用高速公路连接
            # T(x) 控制使用多少变换后的信息 H(x)
            # (1-T(x)) 控制使用多少原始信息 x
            x = H * T + x * (1 - T)
        
        # 转换回原始维度: (batch_size, seq_len, input_dim) -> (batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)
        
        return x

class CBHG(nn.Module): # Tacotron论文中的CBHG模块
    def __init__(self, input_dim, K, conv_channels, num_highway_layers):
        """
        CBHG模块：Conv1D Bank + Highway Network + Bidirectional GRU
        Args:
            input_dim (int): 输入特征维度
            K (int): 1D卷积银行中的卷积核数量
            conv_channels (int): 卷积层的输出通道数
            num_highway_layers (int): 高速公路网络的层数
        """
        super(CBHG, self).__init__()
        self.input_dim = input_dim
        self.K = K
        self.conv_channels = conv_channels
        self.num_highway_layers = num_highway_layers
        
        # Conv1D Bank
        self.conv1d_bank = Conv1DBank(K, input_dim, conv_channels)
        
        # 最大池化层
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
        
        # 投影层 - 将K*conv_channels维度投影回input_dim
        # 按照论文，这里使用两个1x1卷积层
        self.proj_conv1 = nn.Conv1d(K * conv_channels, conv_channels, kernel_size=1)
        self.proj_conv2 = nn.Conv1d(conv_channels, input_dim, kernel_size=1)
        self.proj_bn1 = nn.BatchNorm1d(conv_channels)
        self.proj_bn2 = nn.BatchNorm1d(input_dim)
        
        # 高速公路网络
        self.highway_net = HighWayNet(input_dim, num_highway_layers)
        
        # 双向GRU 
        # RNN天然适合channels_last格式，我们在处理时进行维度转换
        # 保持输入输出为channels_first，中间处理使用channels_last
        self.bidirectional_gru = nn.GRU(
            input_size=input_dim,
            hidden_size=input_dim // 2,
            num_layers=1,
            batch_first=True,  # 使用batch_first=True: (batch_size, seq_len, input_dim)
            bidirectional=True
        )
    
    def forward(self, x):
        """
        Args:
            x (Tensor): 输入张量，形状为 (batch_size, input_dim, seq_len)
        Returns:
            Tensor: 输出张量，形状为 (batch_size, input_dim, seq_len)
        """
        batch_size, input_dim, seq_len = x.size()
        
        # Conv1D Bank
        # 输入: (batch_size, input_dim, seq_len)
        # 输出: (batch_size, K*conv_channels, seq_len)
        conv_out = self.conv1d_bank(x)
        
        # 最大池化
        pooled = self.max_pool(conv_out)
        # 调整长度以匹配原始序列长度
        if pooled.size(2) > seq_len:
            pooled = pooled[:, :, :seq_len]
        
        # 投影层
        # 第一个投影卷积: K*conv_channels -> conv_channels
        proj1 = self.proj_conv1(pooled)
        proj1 = self.proj_bn1(proj1)
        proj1 = F.relu(proj1)
        
        # 第二个投影卷积: conv_channels -> input_dim
        proj2 = self.proj_conv2(proj1)
        proj2 = self.proj_bn2(proj2)
        
        # 残差连接
        # 将投影结果与原始输入相加
        residual = proj2 + x
        
        # 高速公路网络 
        # 输入输出都是: (batch_size, input_dim, seq_len)
        highway_out = self.highway_net(residual)
        
        # 双向GRU处理
        # 从channels_first转换为channels_last: (batch_size, seq_len, input_dim)
        highway_transposed = highway_out.transpose(1, 2)
        
        # GRU前向传播 (使用channels_last格式)
        gru_out, _ = self.bidirectional_gru(highway_transposed)
        
        # 从channels_last转换回channels_first: (batch_size, input_dim, seq_len)  
        gru_out = gru_out.transpose(1, 2)
        
        return gru_out

####### Tacotron 1 的文本编码器部分 #######

class PreNet(nn.Module): # 预处理网络，对嵌入向量进行非线性变换
    def __init__(self, 
                 input_dim,   # 输入特征维度（通常是embedding维度）
                 hidden_dim,  # 隐藏层维度
                 output_dim   # 输出特征维度
                 ):  
        super(PreNet, self).__init__() # 接受一个[batch_size, input_dim]的输入
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        return self.net(x)

class TextEncoder(nn.Module): # 是的孩子们，我不是Transformer
    def __init__(self,
                 # Embedding 参数
                 phoneme_vocab_size,    # 音素词表大小
                 embedding_dim,         # 音素嵌入维度
                 # PreNet 参数  
                 prenet_hidden_dim,     # PreNet隐藏层维度
                 prenet_output_dim,     # PreNet输出维度
                 # CBHG 参数
                 cbhg_k,               # 1D卷积银行中的卷积核数量
                 cbhg_conv_channels,   # 卷积层输出通道数
                 cbhg_highway_layers   # 高速公路网络层数
    ):
        super(TextEncoder, self).__init__()

        # Embedding (从音素ID到向量属性)
        self.phonemembedding = nn.Embedding(
            phoneme_vocab_size, 
            embedding_dim
            )
        # PreNet (向量polish处理)
        self.prenet = PreNet(
            input_dim=embedding_dim,
            hidden_dim=prenet_hidden_dim,
            output_dim=prenet_output_dim
        )
        # CBHG (核心特征提取module)
        self.cbhg = CBHG(
            input_dim=prenet_output_dim,
            K=cbhg_k,
            conv_channels=cbhg_conv_channels,
            num_highway_layers=cbhg_highway_layers
        )
    
    def forward(self, phoneme_ids):
        """
        Args:
            phoneme_ids: "今天是个好天气" -> [71,85,46,99,45,21] # 整数序列
        Returns:
            Tensor: 经过三个组件处理的文本特征
        """

        # Phoneme Embedding: [batch_size, seq_len] (每个bs当中都是一句话被编码为音素序列)
        embedded = self.phonemembedding(phoneme_ids)
        # 编码为 ↓
        # PreNet : [batch_size, seq_len, embedding_dim]
        prenet_out = self.prenet(embedded)
        # 编码为 ↓ 
        prenet_out = prenet_out.transpose(1, 2) # [bs, seq_len, prenet_opt_dim] -> [bs, prenet_opt_dim, seq_len]
        # CBHG : [bs, prenet_opt_dim, seq_len] -> [bs, prenet_opt_dim, seq_len]
        encoded_text_feature = self.cbhg(prenet_out)
        # 编码为 ↓
        return encoded_text_feature.transpose(1, 2) # [bs, seq_len, prenet_opt_dim]
    
####### Tacotron 1 的骨干网络 #######

class AdditiveAttention(nn.Module): # 加性注意力
    def __init__(self, 
                 encoder_dim, 
                 decoder_dim,
                 attention_dim
                 ):
        """
        Args:
            encoder_dim (int): 编码器输出维度
            decoder_dim (int): 解码器输出维度
            attention_dim (int): 注意力维度
        """
        super(AdditiveAttention, self).__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.attention_dim = attention_dim

        # 投影
        self.encoder_proj = nn.Linear(encoder_dim, attention_dim)
        self.decoder_proj = nn.Linear(decoder_dim, attention_dim)
        # 注意力权重计算
        self.v = nn.Linear(attention_dim, 1)
    
    def forward(self, encoder_out, decoder_hidden):
        """
        Args:
            encoder_out (Tensor): 编码器输出, [bs, seq_len, encoder_dim] (TextEncoder的输出格式)
            decoder_hidden (Tensor): 解码器隐藏状态, [bs, decoder_dim]
        Returns:
            context: 上下文向量 [bs, encoder_dim]
            attention_weights: 注意力权重, [bs, seq_len]
        """
        # TextEncoder输出格式: [bs, seq_len, encoder_dim]
        batch_size, seq_len, encoder_dim = encoder_out.size()
        
        # 编码器输出投影: [bs, seq_len, encoder_dim] -> [bs, seq_len, attention_dim]
        encoder_projected = self.encoder_proj(encoder_out)  # [bs, seq_len, attention_dim]
        
        # 解码器隐藏状态投影: [bs, decoder_dim] -> [bs, attention_dim]
        decoder_projected = self.decoder_proj(decoder_hidden)  # [bs, attention_dim]
        
        # 扩展解码器投影以匹配序列长度: [bs, attention_dim] -> [bs, seq_len, attention_dim]
        decoder_projected = decoder_projected.unsqueeze(1).expand(batch_size, seq_len, self.attention_dim)
        
        # 加性注意力计算: tanh(W_e * encoder + W_d * decoder)
        # [bs, seq_len, attention_dim] + [bs, seq_len, attention_dim] -> [bs, seq_len, attention_dim]
        attention_hidden = torch.tanh(encoder_projected + decoder_projected)
        
        # 计算注意力分数: [bs, seq_len, attention_dim] -> [bs, seq_len, 1] -> [bs, seq_len]
        attention_scores = self.v(attention_hidden).squeeze(-1)  # [bs, seq_len]
        
        # 应用softmax获得注意力权重
        attention_weights = torch.softmax(attention_scores, dim=1)  # [bs, seq_len]
        
        # 计算上下文向量: 加权求和
        # encoder_out: [bs, seq_len, encoder_dim]
        # attention_weights: [bs, seq_len] -> [bs, seq_len, 1]
        attention_weights_expanded = attention_weights.unsqueeze(-1)  # [bs, seq_len, 1]
        context = torch.sum(encoder_out * attention_weights_expanded, dim=1)  # [bs, encoder_dim]
        
        return context, attention_weights

class RNN(nn.Module):
    def __init__(self,
                 # 输入维度
                 encoder_dim,           # 编码器输出维度 (与TextEncoder输出一致)
                 attention_dim,         # 注意力机制维度
                 # PreNet 参数
                 prenet_hidden_dim,     # PreNet隐藏层维度  
                 prenet_output_dim,     # PreNet输出维度
                 # RNN 参数
                 decoder_rnn_dim,       # 解码器RNN隐藏层维度
                 # 输出参数
                 num_mels=128,              # mel频谱的维度 和hifigan一致
                 max_decoder_steps=1024 # 最大解码步数
                 ):
        """
        Args:
            encoder_dim: 编码器输出维度
            attention_dim: 注意力机制维度
            prenet_hidden_dim: PreNet隐藏层维度  
            prenet_output_dim: PreNet输出维度
            decoder_rnn_dim: 解码器RNN隐藏层维度
            num_mels: mel频谱维度
            max_decoder_steps: 最大解码步数
        """
        super(RNN, self).__init__()
        
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.decoder_rnn_dim = decoder_rnn_dim
        self.num_mels = num_mels
        self.max_decoder_steps = max_decoder_steps
        
        # PreNet: 对上一帧mel频谱进行预处理
        self.prenet = PreNet(
            input_dim=num_mels,
            hidden_dim=prenet_hidden_dim,
            output_dim=prenet_output_dim
        )
        
        # 注意力机制
        self.attention = AdditiveAttention(
            encoder_dim=encoder_dim,
            decoder_dim=decoder_rnn_dim,
            attention_dim=attention_dim
        )
        
        # GRU
        self.decoder_rnn = nn.GRU(
            input_size=prenet_output_dim + encoder_dim,
            hidden_size=decoder_rnn_dim,
            batch_first=True
        )
        
        # 输出投影层: RNN输出 + context -> mel频谱
        self.output_projection = nn.Linear(
            decoder_rnn_dim + encoder_dim, 
            num_mels
        )
        
        # 停止标记预测
        self.stop_projection = nn.Linear(
            decoder_rnn_dim + encoder_dim,
            1
        )
    
    def forward(self, encoder_outputs, target_mels=None):
        """
        Args:
            encoder_outputs: TextEncoder的输出 [bs, seq_len, encoder_dim]
            target_mels: 训练时的目标mel频谱 [bs, target_len, num_mels], 推理时为None
        Returns:
            mel_outputs: 预测的mel频谱 [bs, output_len, num_mels]
            stop_tokens: 停止标记 [bs, output_len]
            attention_weights: 注意力权重 [bs, output_len, seq_len]
        """
        batch_size = encoder_outputs.size(0)
        
        if target_mels is not None:
            # 训练模式: Teacher Forcing
            return self._forward_train(encoder_outputs, target_mels)
        else:
            # 推理模式: 自回归生成
            return self._forward_inference(encoder_outputs, batch_size)
    
    def _forward_train(self, encoder_outputs, target_mels):
        batch_size, target_len, _ = target_mels.size()
        
        # 初始化
        decoder_hidden = self._init_decoder_hidden(batch_size)
        
        # 存储输出
        mel_outputs = []
        stop_tokens = []
        attention_weights_list = []
        
        # 第一帧使用全零输入
        prev_mel = torch.zeros(batch_size, self.num_mels, device=encoder_outputs.device)
        
        for t in range(target_len):
            # PreNet处理
            prenet_out = self.prenet(prev_mel)  # [bs, prenet_output_dim]
            
            # 注意力机制
            context, attention_weights = self.attention(encoder_outputs, decoder_hidden.squeeze(0))
            
            # RNN输入: prenet_out + context
            rnn_input = torch.cat([prenet_out, context], dim=-1).unsqueeze(1)  # [bs, 1, prenet_output_dim + encoder_dim]
            
            # RNN前向传播
            rnn_output, decoder_hidden = self.decoder_rnn(rnn_input, decoder_hidden)
            rnn_output = rnn_output.squeeze(1)  # [bs, decoder_rnn_dim]
            
            # 输出投影
            decoder_output = torch.cat([rnn_output, context], dim=-1)  # [bs, decoder_rnn_dim + encoder_dim]
            mel_output = self.output_projection(decoder_output)  # [bs, num_mels]
            stop_logit = self.stop_projection(decoder_output).squeeze(-1)  # [bs]
            
            # 存储输出
            mel_outputs.append(mel_output)
            stop_tokens.append(stop_logit)
            attention_weights_list.append(attention_weights)
            
            # Teacher Forcing: 使用真实的目标作为下一步输入
            prev_mel = target_mels[:, t, :]
        
        # 转换为张量
        mel_outputs = torch.stack(mel_outputs, dim=1)  # [bs, target_len, num_mels]
        stop_tokens = torch.stack(stop_tokens, dim=1)  # [bs, target_len]
        attention_weights = torch.stack(attention_weights_list, dim=1)  # [bs, target_len, seq_len]
        
        return mel_outputs, stop_tokens, attention_weights
    
    def _forward_inference(self, encoder_outputs, batch_size):
        # 初始化
        decoder_hidden = self._init_decoder_hidden(batch_size)
        
        # 存储输出
        mel_outputs = []
        stop_tokens = []
        attention_weights_list = []
        
        # 第一帧使用全零输入
        prev_mel = torch.zeros(batch_size, self.num_mels, device=encoder_outputs.device)
        
        for t in range(self.max_decoder_steps):
            # PreNet处理
            prenet_out = self.prenet(prev_mel)  # [bs, prenet_output_dim]
            
            # 注意力机制
            context, attention_weights = self.attention(encoder_outputs, decoder_hidden.squeeze(0))
            
            # RNN输入: prenet_out + context
            rnn_input = torch.cat([prenet_out, context], dim=-1).unsqueeze(1)  # [bs, 1, prenet_output_dim + encoder_dim]
            
            # RNN前向传播
            rnn_output, decoder_hidden = self.decoder_rnn(rnn_input, decoder_hidden)
            rnn_output = rnn_output.squeeze(1)  # [bs, decoder_rnn_dim]
            
            # 输出投影
            decoder_output = torch.cat([rnn_output, context], dim=-1)  # [bs, decoder_rnn_dim + encoder_dim]
            mel_output = self.output_projection(decoder_output)  # [bs, num_mels]
            stop_logit = self.stop_projection(decoder_output).squeeze(-1)
            # 存储输出
            mel_outputs.append(mel_output)
            stop_tokens.append(stop_logit)
            attention_weights_list.append(attention_weights)
            
            # 使用预测的mel作为下一步输入
            prev_mel = mel_output
            
            # 检查停止条件 (如果所有样本的停止概率都大于0.5)
            if torch.all(torch.sigmoid(stop_logit) > 0.5): # 主要是方便后面BCELoss的计算
                break
        
        # 转换为张量
        mel_outputs = torch.stack(mel_outputs, dim=1)  # [bs, output_len, num_mels]
        stop_tokens = torch.stack(stop_tokens, dim=1)  # [bs, output_len]
        attention_weights = torch.stack(attention_weights_list, dim=1)  # [bs, output_len, seq_len]
        
        return mel_outputs, stop_tokens, attention_weights
    
    def _init_decoder_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.decoder_rnn_dim, 
                          device=next(self.parameters()).device)

####### Tacotron 1 #######

class Tacotron1(nn.Module):
    def __init__(self,
                 # TextEncoder 参数
                 phoneme_vocab_size,    # 音素词表大小
                 embedding_dim=256,     # 音素嵌入维度
                 encoder_prenet_hidden_dim=256,
                 encoder_prenet_output_dim=128,
                 cbhg_k=16,
                 cbhg_conv_channels=128,
                 cbhg_highway_layers=4,
                 # 注意力参数
                 attention_dim=128,
                 # Decoder 参数
                 decoder_prenet_hidden_dim=256,
                 decoder_prenet_output_dim=128,
                 decoder_rnn_dim=256,
                 # 输出参数 (与HiFiGAN兼容)
                 num_mels=80,          # mel频谱维度，与HiFiGAN配置一致
                 max_decoder_steps=1024
                 ):
        """        
        Args:
            phoneme_vocab_size: 音素词表大小
            embedding_dim: 音素嵌入维度
            encoder_prenet_hidden_dim: 编码器PreNet隐藏层维度
            encoder_prenet_output_dim: 编码器PreNet输出维度 (也是TextEncoder最终输出维度)
            cbhg_k: CBHG中1D卷积银行的卷积核数量
            cbhg_conv_channels: CBHG卷积层输出通道数
            cbhg_highway_layers: CBHG高速公路网络层数
            attention_dim: 注意力机制维度
            decoder_prenet_hidden_dim: 解码器PreNet隐藏层维度
            decoder_prenet_output_dim: 解码器PreNet输出维度
            decoder_rnn_dim: 解码器RNN隐藏层维度
            num_mels: mel频谱维度 (128维，与HiFiGAN配置一致)
            max_decoder_steps: 最大解码步数
        """
        super(Tacotron1, self).__init__()
        
        # 文本编码器
        # 输入: [batch_size, text_seq_len] (音素ID序列)
        # 输出: [batch_size, text_seq_len, encoder_prenet_output_dim]
        self.text_encoder = TextEncoder(
            phoneme_vocab_size=phoneme_vocab_size,
            embedding_dim=embedding_dim,
            prenet_hidden_dim=encoder_prenet_hidden_dim,
            prenet_output_dim=encoder_prenet_output_dim,
            cbhg_k=cbhg_k,
            cbhg_conv_channels=cbhg_conv_channels,
            cbhg_highway_layers=cbhg_highway_layers
        )
        
        # RNN解码器 (encoder_prenet_output_dim是TextEncoder的最终输出维度)
        # 输入: [batch_size, text_seq_len, encoder_prenet_output_dim] + 可选的target_mels
        # 输出: mel频谱 [batch_size, mel_seq_len, num_mels]
        self.decoder = RNN(
            encoder_dim=encoder_prenet_output_dim,  # TextEncoder的输出维度
            attention_dim=attention_dim,
            prenet_hidden_dim=decoder_prenet_hidden_dim,
            prenet_output_dim=decoder_prenet_output_dim,
            decoder_rnn_dim=decoder_rnn_dim,
            num_mels=num_mels,
            max_decoder_steps=max_decoder_steps
        )
        
        # 存储配置参数
        self.num_mels = num_mels
        self.max_decoder_steps = max_decoder_steps
    
    def forward(self, phoneme_ids, target_mels=None):
        """
        Args:
            phoneme_ids: 音素ID序列
                形状: [batch_size, text_seq_len]
                内容: 整数序列，每个整数代表一个音素的ID
                
            target_mels: 训练时的目标mel频谱 (可选)
                形状: [batch_size, mel_seq_len, num_mels] 或 None
                内容: 目标mel频谱，训练时用于Teacher Forcing，推理时为None
        Returns:
            dict: 包含以下键值对
                - mel_outputs: 预测的mel频谱
                    形状: [batch_size, mel_seq_len, num_mels=128]
                    内容: 直接用于HiFiGAN声码器的mel频谱
                - stop_tokens: 停止标记logits
                    形状: [batch_size, mel_seq_len]
                    内容: 每个时间步的停止概率logits (未经sigmoid)
                - attention_weights: 注意力权重
                    形状: [batch_size, mel_seq_len, text_seq_len]
                    内容: 解码器每个时间步对编码器各位置的注意力权重
        """

        # 文本编码阶段
        # 输入: phoneme_ids [batch_size, text_seq_len]
        # 输出: encoder_outputs [batch_size, text_seq_len, encoder_dim=128]
        encoder_outputs = self.text_encoder(phoneme_ids)
        
        # 解码阶段 - 生成mel频谱
        # 输入: encoder_outputs [batch_size, text_seq_len, encoder_dim=128]
        #       target_mels [batch_size, mel_seq_len, num_mels=128] 或 None
        # 输出: mel_outputs [batch_size, mel_seq_len, num_mels=128]
        #       stop_tokens [batch_size, mel_seq_len] (logits)
        #       attention_weights [batch_size, mel_seq_len, text_seq_len]
        mel_outputs, stop_tokens, attention_weights = self.decoder(
            encoder_outputs, target_mels
        )
        
        return {
            'mel_outputs': mel_outputs,      # [batch_size, mel_seq_len, 128] - 直接给HiFiGAN
            'stop_tokens': stop_tokens,      # [batch_size, mel_seq_len] - 用于训练的停止损失
            'attention_weights': attention_weights  # [batch_size, mel_seq_len, text_seq_len] - 可视化用
        }
    
    def inference(self, phoneme_ids):
        """
        Args:
            phoneme_ids: 音素ID序列
                形状: [batch_size, text_seq_len]
                内容: 待合成文本的音素ID序列
        Returns:
            tuple: (mel_outputs, attention_weights)
                mel_outputs: 生成的mel频谱
                    形状: [batch_size, generated_mel_seq_len, 128]
                    内容: 可直接输入HiFiGAN进行音频合成
                attention_weights: 注意力权重矩阵
                    形状: [batch_size, generated_mel_seq_len, text_seq_len] 
                    内容: 用于分析对齐质量和可视化
        """
        with torch.no_grad():
            # 调用forward，target_mels=None触发推理模式
            outputs = self.forward(phoneme_ids, target_mels=None)
            
            # 返回HiFiGAN需要的mel频谱和注意力权重
            return outputs['mel_outputs'], outputs['attention_weights']
    
    def get_model_info(self):
        """
        获取模型配置信息
        
        Returns:
            dict: 模型配置信息
        """
        return {
            'model_type': 'Tacotron1',
            'mel_channels': self.num_mels,  # 128维，与HiFiGAN兼容
            'max_decoder_steps': self.max_decoder_steps,
            'vocoder_compatible': 'HiFiGAN',
            'architecture_note': '移除PostNet，直接输出mel频谱给HiFiGAN'
        }
