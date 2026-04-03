from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from config import get_rqvae_config, get_rope_config, get_embedding_config
from config import get_semantic_id_config
from dataset import save_emb
from config import get_time_interval_config

class RotaryPositionalEmbedding(torch.nn.Module):
    """
    RoPE (Rotary Position Embedding) 实现
    将位置信息通过旋转编码的方式注入到 Q, K 中
    """
    def __init__(self, head_dim, max_seq_len=512, theta=10000.0, device=None):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # 确保head_dim是偶数
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        
        # 预计算旋转矩阵的频率
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # 预计算cos和sin缓存
        self._build_cache(max_seq_len, device)
    
    def _build_cache(self, seq_len, device=None):
        """构建cos和sin的缓存 - 优化混合精度训练"""
        # 仅在需要时重建；缓存为半精度，节省显存
        if (hasattr(self, 'cos_cache') and self.cos_cache is not None and 
            self.cos_cache.shape[0] >= seq_len):
            return
            
        if device is None:
            device = self.inv_freq.device
            
        seq_idx = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(seq_idx, self.inv_freq)  # [seq_len, head_dim//2]
        
        # 扩展到完整维度 [seq_len, head_dim]
        cos = torch.cos(freqs).repeat_interleave(2, dim=-1)
        sin = torch.sin(freqs).repeat_interleave(2, dim=-1)
        
        # 🎯 缓存为半精度，节省显存并兼容混合精度训练
        self.register_buffer('cos_cache', cos.to(torch.float16), persistent=False)
        self.register_buffer('sin_cache', sin.to(torch.float16), persistent=False)
        self.cached_seq_len = seq_len
    
    def forward(self, q, k, seq_len=None):
        """
        对 Q 和 K 应用 RoPE
        
        Args:
            q, k: [batch_size, num_heads, seq_len, head_dim]
            seq_len: 序列长度，如果超过缓存长度会重新构建缓存
            
        Returns:
            rotated_q, rotated_k: 应用RoPE后的Q和K
        """
        if seq_len is None:
            seq_len = q.size(2)
            
        # 如果序列长度超过缓存，重新构建缓存
        if seq_len > self.cached_seq_len:
            self._build_cache(seq_len, q.device)
        
        # 获取当前序列长度的cos和sin
        cos = self.cos_cache[:seq_len]  # [seq_len, head_dim]
        sin = self.sin_cache[:seq_len]  # [seq_len, head_dim]
        
        # 🎯 确保与当前计算 dtype 对齐，兼容混合精度训练
        cos = cos.to(dtype=q.dtype)
        sin = sin.to(dtype=q.dtype)
        
        # 应用旋转
        rotated_q = self._apply_rotary_emb(q, cos, sin)
        rotated_k = self._apply_rotary_emb(k, cos, sin)
        
        return rotated_q, rotated_k
    
    def _apply_rotary_emb(self, x, cos, sin):
        """
        应用旋转嵌入
        x: [batch_size, num_heads, seq_len, head_dim]
        cos, sin: [seq_len, head_dim]
        """
        # 分离奇偶位置
        x1 = x[..., 0::2]  # [batch_size, num_heads, seq_len, head_dim//2]
        x2 = x[..., 1::2]  # [batch_size, num_heads, seq_len, head_dim//2]
        
        # 应用旋转公式
        cos = cos[..., 0::2]  # [seq_len, head_dim//2]
        sin = sin[..., 1::2]  # [seq_len, head_dim//2]
        
        # 旋转变换
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos
        
        # 重新交错排列
        rotated_x = torch.stack([rotated_x1, rotated_x2], dim=-1).flatten(-2)
        
        return rotated_x

def create_candidate_head(input_dim, output_dim, head_type='mlp', dropout_rate=0.15):
    """
    🎯 创建候选侧头部网络 - 支持多种类型以优化检索性能
    
    Args:
        input_dim: 输入维度
        output_dim: 输出维度  
        head_type: 头部类型 ('mlp', 'linear', 'identity', 'light_mlp')
        dropout_rate: dropout率
        
    Returns:
        torch.nn.Module: 对应的头部网络
    """
    if head_type == 'linear':
        # 🔥 推荐：简单线性头，适合大规模检索
        return torch.nn.Linear(input_dim, output_dim, bias=True)
    
    elif head_type == 'identity':
        # 🔧 零开销：维度对齐时使用
        if input_dim == output_dim:
            return torch.nn.Identity()
        else:
            # 维度不对齐时降级为线性
            print(f"⚠️  维度不对齐({input_dim}!={output_dim})，Identity降级为Linear")
            return torch.nn.Linear(input_dim, output_dim, bias=True)
    
    elif head_type == 'light_mlp':
        # 🎯 轻量MLP：无残差、低dropout
        hidden_dim = max(output_dim * 2, 64)  # 较小的扩展因子
        return torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout_rate * 0.5),  # 减半的dropout
            torch.nn.Linear(hidden_dim, output_dim)
        )
    
    elif head_type == 'mlp':
        # 📊 传统：使用EnhancedDNN
        return EnhancedDNN(input_dim, output_dim, MLP_dropout_rate=dropout_rate)
    
    else:
        raise ValueError(f"不支持的候选头部类型: {head_type}")


class FlashMultiHeadAttention(torch.nn.Module):
    def __init__(self, hidden_units, num_heads, transformer_dropout, enable_rope=False, enable_time_features=False, enable_time_bias=False, attention_mode='sdpa', enable_relative_bias=False, use_checkpoint=False, hstu_match_official=True, hstu_length_norm_mode='N'):
        super(FlashMultiHeadAttention, self).__init__()

        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads
        self.dropout_rate = transformer_dropout
        self.enable_rope = enable_rope
        self.enable_time_features = enable_time_features
        self.enable_time_bias = enable_time_bias
        self.attention_mode = attention_mode  # 'sdpa', 'softmax', 'hstu'
        self.enable_relative_bias = enable_relative_bias
        self.use_checkpoint = use_checkpoint  # 🎯 梯度检查点优化
        
        self.hstu_match_official = hstu_match_official
        self.hstu_length_norm_mode = hstu_length_norm_mode  # 'none' | 'L' | 'sqrtL' | 'N'

        assert hidden_units % num_heads == 0, "hidden_units must be divisible by num_heads"
        assert attention_mode in ['sdpa', 'softmax', 'hstu'], f"Unsupported attention_mode: {attention_mode}"

        self.q_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.k_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.v_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.out_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.dropout = torch.nn.Dropout(self.dropout_rate)
        # 🎯 RoPE位置编码
        if self.enable_rope:
            rope_config = get_rope_config()
            self.rope = RotaryPositionalEmbedding(
                head_dim=self.head_dim,
                max_seq_len=rope_config['rope_max_seq_len'],
                theta=rope_config['rope_theta']
            )
        
        if self.enable_time_bias:
            # 🎯 正约束的时间衰减参数（使用raw参数+softplus确保正值）
            self.time_decay_gamma_raw = torch.nn.Parameter(torch.ones(num_heads) * 0.1)
            # 🎯 多尺度时间参数可训练（分钟/小时/天/周）
            self.time_scale_tau = torch.nn.Parameter(torch.tensor([1/60, 1.0, 24.0, 168.0]))  # 分钟、小时、天、周
            # 🎯 多尺度权重（per-head softmax归一化，初值等权）
            self.time_scale_weights_raw = torch.nn.Parameter(torch.zeros(num_heads, 4))  # [H, 4]
            # 🎯 EMA分位数统计，用于尺度归一化
            self.register_buffer('time_delta_ema_q90', torch.tensor(24.0))  # EMA of q90 (hours)
            self.register_buffer('ema_momentum', torch.tensor(0.99))  # EMA更新动量
            # 🎯 上界控制参数
            self.time_bias_clip_value = 6.0
            
            # 🚀 新增：动作类型感知门控开关与参数（默认启用）
            self.enable_action_type_time_gating = False
            # 动作类型：0=PAD, 1=曝光, 2=点击 → 共3类（与dataset中非PAD=2一致）
            self.num_token_types_for_gating = 3
            
            # 形状：[H, T, T]，对不同Head+动作对赋予不同缩放（0.5~1.5）
            self.action_pair_gate_raw = torch.nn.Parameter(
                torch.zeros(self.num_heads, self.num_token_types_for_gating, self.num_token_types_for_gating)
            )
            
            # 合理初始化：保持稳定并微弱偏向点击-点击更强
            with torch.no_grad():
                # 目标缩放：1.0 → raw=0；1.2 → raw≈0.847；0.9 → raw≈-0.405
                self.action_pair_gate_raw.zero_()
                # expo-expo(1,1) 轻微>1 可不设也稳定；这里保持1.0（raw=0）
                # click-click(2,2) 稍强一些（1.2）
                self.action_pair_gate_raw[:, 2, 2].fill_(0.847)
                # cross pair(1,2)/(2,1) 稍弱一些（0.9）
                self.action_pair_gate_raw[:, 1, 2].fill_(-0.405)
                self.action_pair_gate_raw[:, 2, 1].fill_(-0.405)
                # 与PAD交互的缩放保持1.0（raw=0），实际也会被mask掉
            
            # 🚀 新增：q90 EMA 冷启动安全下限（默认60秒，可按需调大）
            self.register_buffer("time_delta_q90_min", torch.tensor(60.0 / 3600.0, dtype=torch.float32))  # 转换为小时
        
        # 🎯 HSTU pointwise attention参数
        if self.attention_mode == 'hstu':
            # per-head可学习缩放因子，用于控制attention输出的幅度
            self.hstu_head_scale = torch.nn.Parameter(torch.ones(num_heads))
            # 可学习的温度参数，用于控制SiLU激活前的缩放（改为per-head）
            self.hstu_temperature = torch.nn.Parameter(torch.ones(num_heads) / (self.head_dim ** 0.5))
            # 🎯 Q/K 归一化层，稳定点积幅度
            self.hstu_q_norm = torch.nn.LayerNorm(self.head_dim, eps=1e-5)
            self.hstu_k_norm = torch.nn.LayerNorm(self.head_dim, eps=1e-5)
            # 🎯 非负性约束参数：控制是否启用Softplus约束
            self.hstu_use_nonnegative = True  # 可配置开关
            # 🎯 注意力权重监控器（调试用）
            self.register_buffer('_step_counter', torch.tensor(0))
            self.register_buffer('_negative_weight_ratio', torch.tensor(0.0))
            
            # === 官方对齐配置 ===
            # 是否对 Q/K 进行 LN（官方不做）
            self.hstu_use_qk_norm = not hstu_match_official
            # 固定温度 α=1/√d（官方）
            self.register_buffer('_hstu_fixed_alpha', torch.tensor(1.0 / (self.head_dim ** 0.5)))
            
            if self.hstu_match_official:
                # 固定温度，不训练；禁用非负约束；长度归一走 'N'
                self.hstu_use_nonnegative = False
                self.hstu_length_norm_mode = 'N'
                # 冻结 head_scale=1
                with torch.no_grad():
                    self.hstu_head_scale.fill_(1.0)
                self.hstu_head_scale.requires_grad_(False)

        # 🔧 添加注意力机制的权重初始化
        self._init_weights()

    def _init_weights(self):
        """
        为注意力机制使用标准的Xavier初始化
        这对Transformer性能很重要
        """
        # 对Q、K、V、输出投影使用Xavier uniform初始化
        for module in [self.q_linear, self.k_linear, self.v_linear, self.out_linear]:
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        
        # 时间偏置参数使用小的初始值，避免一开始时间偏置过强
        if self.enable_time_bias:
            # 🎯 修正：gamma初始化为真正的小值（0.02-0.08），使用softplus逆变换
            target_gamma_min, target_gamma_max = 0.02, 0.08
            raw_min = torch.log(torch.expm1(torch.tensor(target_gamma_min)) + 1e-6)
            raw_max = torch.log(torch.expm1(torch.tensor(target_gamma_max)) + 1e-6)
            torch.nn.init.uniform_(self.time_decay_gamma_raw, raw_min.item(), raw_max.item())
            # 🎯 time_scale_tau使用softplus参数化避免无效值
            with torch.no_grad():
                self.time_scale_tau.data = torch.log(torch.expm1(self.time_scale_tau.data) + 1e-6)
            # 🎯 多尺度权重初始化为等权（zeros经过softmax后等权）
            torch.nn.init.zeros_(self.time_scale_weights_raw)
        
        # 注意：不再初始化相对位置偏置参数
        
        # 🎯 HSTU pointwise attention参数初始化
        if self.attention_mode == 'hstu':
            torch.nn.init.ones_(self.hstu_head_scale)
            torch.nn.init.constant_(self.hstu_temperature, 1.0 / (self.head_dim ** 0.5))
            # Q/K 归一化层会自动初始化，无需手动设置

    def forward(self, query, key, value, attn_mask=None, seq_timestamps=None, token_type=None, ts_valid_mask=None):
        batch_size, seq_len, _ = query.size()

        # 计算Q, K, V
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)
        
        # 🎯 获取计算 dtype，确保后续操作与混合精度训练兼容
        qkv_dtype = Q.dtype

        # reshape为multi-head格式
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 🎯 Stage 1: 应用RoPE位置编码
        if self.enable_rope:
            Q, K = self.rope(Q, K, seq_len)

        # 🕐 Stage 2: 计算成对时间偏置矩阵
        time_bias = None
        if self.enable_time_bias and seq_timestamps is not None:
            time_bias = self._compute_pairwise_time_bias(seq_timestamps, token_type=token_type, ts_valid_mask=ts_valid_mask)
            # 🎯 确保与计算 dtype 对齐
            if time_bias is not None:
                time_bias = time_bias.to(dtype=qkv_dtype)
        
        # 🎯 Stage 3:（跳过 RAB）relative_pos_bias = None
        # relative_pos_bias = None  # 不使用RAB
        
        # 🎯 Stage 4: 合并偏置为rab_total（此处仅 time_bias）
        rab_total = time_bias
        
        # 🎯 官方对齐：对 bias 做 per-query 零均值，避免整体漂移
        if rab_total is not None and getattr(self, 'hstu_match_official', False) and self.attention_mode == 'hstu':
            rab_total = rab_total - rab_total.mean(dim=-1, keepdim=True)  # [B,H,L,L] 对每个 (b,h,i,:) 去均值

        # 🎯 确保 attn_mask 使用 bool dtype，节省带宽
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(torch.bool)

        # 🎯 Stage 5: 注意力计算核心函数 - 支持激活检查点
        def attn_core(Q, K, V, attn_mask, rab_total):
            if self.attention_mode == 'hstu':
                return self._compute_hstu_attention(Q, K, V, attn_mask, rab_total, batch_size, seq_len)
            elif self.attention_mode == 'sdpa' and hasattr(F, 'scaled_dot_product_attention'):
                return self._compute_sdpa_attention(Q, K, V, attn_mask, rab_total)
            else:
                return self._compute_softmax_attention(Q, K, V, attn_mask, rab_total)

        # 🔄 使用激活检查点（非重入模式，兼容 autocast）
        if self.training and self.use_checkpoint:
            # 为空张量提供默认值以避免检查点函数调用问题
            empty_mask = torch.tensor([], device=Q.device, dtype=torch.bool)
            empty_bias = torch.tensor([], device=Q.device, dtype=qkv_dtype)
            attn_output = checkpoint(
                attn_core, 
                Q, K, V, 
                attn_mask if attn_mask is not None else empty_mask,
                rab_total if rab_total is not None else empty_bias,
                use_reentrant=False
            )
        else:
            attn_output = attn_core(Q, K, V, attn_mask, rab_total)

        # reshape回原来的格式
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_units)

        # 最终的线性变换
        output = self.out_linear(attn_output)
        output = self.dropout(output)  # 添加Dropout
        return output, None
    
    def _hstu_attention_core(self, Q, K, V, attn_mask, rab_total, batch_size, seq_len):
        """
        HSTU注意力核心计算（可被checkpoint包装）
        显存优化：不返回attention_weights，及时释放中间张量
        """
        # 🎯 Step 1: Q/K 归一化（可配置）
        if getattr(self, 'hstu_use_qk_norm', True):
            Q_in = self.hstu_q_norm(Q)
            K_in = self.hstu_k_norm(K)
        else:
            Q_in, K_in = Q, K
        
        # 🎯 Step 2: 温度（官方固定 α=1/√d）
        if getattr(self, 'hstu_match_official', False):
            temperature = self._hstu_fixed_alpha.view(1, 1, 1, 1).expand(1, self.num_heads, 1, 1)
        else:
            temperature = self.hstu_temperature.view(1, -1, 1, 1)
        content_scores = torch.matmul(Q_in, K_in.transpose(-2, -1)) * temperature
        
        # 🎯 Step 3: 加上 time_bias（已在 forward 里做了门控/去偏）
        pre_activation_scores = content_scores + rab_total if rab_total is not None else content_scores
        
        # 🎯 Step 4: 激活 —— 官方用 SiLU（不做非负Softplus）
        if getattr(self, 'hstu_use_nonnegative', False):
            attention_weights = F.softplus(F.silu(pre_activation_scores), beta=1.0)
        else:
            attention_weights = F.silu(pre_activation_scores)
        
        # 🎯 Step 5: 应用掩码：直接将被mask的位置置零（而非-inf）
        if attn_mask is not None:
            # attn_mask: [B, L, L], True表示允许注意
            mask_expanded = attn_mask.unsqueeze(1)  # [B, 1, L, L]
            attention_weights = attention_weights * mask_expanded.float()
        
        # 🎯 Step 6: 长度归一（官方常数缩放）
        mode = getattr(self, 'hstu_length_norm_mode', 'N')
        if mode == 'none':
            length_norm = 1.0
        elif mode == 'N':
            length_norm = torch.tensor(float(seq_len), dtype=attention_weights.dtype, device=attention_weights.device)
        elif mode == 'L':
            if attn_mask is None:
                length_norm = torch.tensor(float(seq_len), dtype=attention_weights.dtype, device=attention_weights.device).view(1,1,1,1)
            else:
                length_norm = attn_mask.float().sum(dim=-1, keepdim=True).clamp(min=1.0).unsqueeze(1)
        elif mode == 'sqrtL':
            if attn_mask is None:
                length_norm = torch.tensor(float(seq_len), dtype=attention_weights.dtype, device=attention_weights.device).sqrt().view(1,1,1,1)
            else:
                length_norm = torch.sqrt(attn_mask.float().sum(dim=-1, keepdim=True).clamp(min=1.0)).unsqueeze(1)
        else:
            raise ValueError(f"Unknown hstu_length_norm_mode={mode}")
        
        # 🎯 Step 7: Dropout（官方通常不做；这里保持开关）
        if self.training and self.dropout_rate > 0 and not getattr(self, 'hstu_match_official', False):
            attention_weights = F.dropout(attention_weights, p=self.dropout_rate, training=True)
        
        # 🎯 Step 8: 聚合
        attn_output = torch.matmul(attention_weights, V) / length_norm
        
        # 释放中间
        del attention_weights, pre_activation_scores, content_scores
        
        # 🎯 Step 9: head_scale（官方可关）
        if not getattr(self, 'hstu_match_official', False):
            attn_output = attn_output * self.hstu_head_scale.view(1, -1, 1, 1)
        
        return attn_output
    
    def _compute_hstu_attention(self, Q, K, V, attn_mask, rab_total, batch_size, seq_len):
        """
        HSTU-style pointwise SiLU attention with improvements
        使用SiLU激活替代softmax，保留用户偏好强度信息
        
        主要改进：
        1. Q/K 归一化稳定点积幅度
        2. 可选非负性约束避免负权重抵消
        3. 权重监控用于调试分析
        4. 🎯 梯度检查点优化（可选）
        5. 显存优化：不保存attention_weights
        """
        # 🎯 梯度检查点优化：减少显存占用
        # if self.use_checkpoint and self.training:
        if self.training:
            # 使用checkpoint包装核心计算，减少中间激活的显存占用
            attn_output = checkpoint(
                self._hstu_attention_core,
                Q, K, V, attn_mask, rab_total, batch_size, seq_len,
                use_reentrant=False  # 使用新版checkpoint API
            )
        else:
            # 正常计算路径
            attn_output = self._hstu_attention_core(
                Q, K, V, attn_mask, rab_total, batch_size, seq_len
            )
        
        return attn_output
    
    def _compute_sdpa_attention(self, Q, K, V, attn_mask, rab_total):
        """
        PyTorch 2.0+ SDPA attention with bias support
        """
        # 🔧 修正掩码语义：SDPA需要additive mask (0=允许, -inf=禁止)，而非bool mask
        if attn_mask is not None:
            # 转换bool mask (True=允许) 为 additive mask (0=允许, -inf=禁止)
            additive_mask = torch.zeros_like(attn_mask, dtype=Q.dtype, device=Q.device)
            additive_mask.masked_fill_(attn_mask.logical_not(), float('-inf'))
            
            # 融合rab_total到掩码中
            if rab_total is not None:
                additive_mask = additive_mask.unsqueeze(1) + rab_total  # [B, H, L, L]
            else:
                additive_mask = additive_mask.unsqueeze(1)  # [B, 1, L, L] -> broadcast
                
            attn_output = F.scaled_dot_product_attention(
                Q, K, V, dropout_p=self.dropout_rate if self.training else 0.0, attn_mask=additive_mask
            )
        else:
            # 仅有rab_total的情况
            attn_output = F.scaled_dot_product_attention(
                Q, K, V, dropout_p=self.dropout_rate if self.training else 0.0, 
                attn_mask=rab_total if rab_total is not None else None
            )
        
        return attn_output
    
    def _compute_softmax_attention(self, Q, K, V, attn_mask, rab_total):
        """
        标准softmax attention with bias support
        """
        scale = (self.head_dim) ** -0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

        # 应用掩码
        if attn_mask is not None:
            scores.masked_fill_(attn_mask.unsqueeze(1).logical_not(), float('-inf'))
        
        # 应用rab_total偏置
        if rab_total is not None:
            scores = scores + rab_total

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout_rate, training=self.training) 
        attn_output = torch.matmul(attn_weights, V)
        
        return attn_output
    
    # 注意：已删除相对位置偏置相关方法，仅保留 RoPE + time_bias

    def _compute_pairwise_time_bias(self, seq_timestamps, token_type=None, ts_valid_mask=None):
        """
        计算成对时间偏置矩阵 [B, H, L, L]
        🎯 优化版本：移除冗余压缩，使用正约束gamma，多尺度softmax权重，tanh上界控制
        """
        # seq_timestamps: [B, L] Unix时间戳
        # 计算成对时间差矩阵 [B, L, L]
        time_i = seq_timestamps.unsqueeze(2)  # [B, L, 1]
        time_j = seq_timestamps.unsqueeze(1)  # [B, 1, L]
        time_delta_seconds = torch.abs(time_i - time_j).float()  # [B, L, L]
        
        # 转换为小时
        time_delta_hours = time_delta_seconds / 3600.0  # [B, L, L]
        
        # 🎯 EMA分位数缩放：训练时更新EMA，推理时使用固定EMA值
        if self.training:
            with torch.no_grad():
                valid_deltas = time_delta_hours[time_delta_hours > 0]  # 排除对角线0值
                if len(valid_deltas) > 0:
                    current_q90 = torch.quantile(valid_deltas, 0.9)
                    # EMA更新
                    self.time_delta_ema_q90.copy_(
                        self.ema_momentum * self.time_delta_ema_q90 + 
                        (1 - self.ema_momentum) * current_q90
                    )
        
        # 🚀 使用 max(EMA_q90, q90_min) 做归一化分母，避免训练初期过小导致不稳定
        q90_denom = torch.clamp(self.time_delta_ema_q90, min=self.time_delta_q90_min)
        time_delta_normalized = time_delta_hours / (q90_denom + 1e-6)
        
        # 🎯 多尺度时间偏置计算（仅使用log1p，移除asinh冗余压缩）
        tau_scales = F.softplus(self.time_scale_tau).view(1, 1, 1, -1) + 1e-6  # [1, 1, 1, 4]
        delta_normalized = time_delta_normalized.unsqueeze(-1) / tau_scales  # [B, L, L, 4]
        
        # 对每个尺度计算log衰减
        time_bias_per_scale = -torch.log1p(delta_normalized)  # [B, L, L, 4]
        
        # 🎯 使用per-head softmax权重进行多尺度加权
        scale_weights = F.softmax(self.time_scale_weights_raw, dim=-1)  # [H, 4]
        # 修复einsum下标：两个长度维应使用不同标记，且输出下标不得重复
        time_bias_multi_scale = torch.einsum('bijk,hk->bhij', time_bias_per_scale, scale_weights)  # [B, H, L, L]
        
        # 🎯 应用正约束的per-head衰减系数
        time_decay_gamma = F.softplus(self.time_decay_gamma_raw) + 1e-6  # [H]
        time_bias = time_bias_multi_scale * time_decay_gamma.view(1, -1, 1, 1)  # [B, H, L, L]
        
        # 🎯 tanh上界控制，防止极端值
        time_bias = self.time_bias_clip_value * torch.tanh(time_bias / self.time_bias_clip_value)
        
        # 🚀 新增：动作类型感知的时间bias门控（按动作对缩放时间偏置强度）
        if (token_type is not None) and self.enable_action_type_time_gating:
            # token_type: (B, L) 0=PAD, 1=曝光, 2=点击
            # 构造 one-hot，(B, L, 3)
            num_types = self.num_token_types_for_gating
            q_types = token_type  # 自注意，Query/Key同源
            k_types = token_type
            q_onehot = torch.nn.functional.one_hot(q_types.clamp(0, num_types - 1).long(), num_classes=num_types).float()
            k_onehot = torch.nn.functional.one_hot(k_types.clamp(0, num_types - 1).long(), num_classes=num_types).float()
            
            # 计算缩放矩阵（0.5~1.5）：0.5 + sigmoid(raw)
            gate_T = 0.5 + torch.sigmoid(self.action_pair_gate_raw)  # (H, 3, 3)
            
            # 将 (B, Lq, 3) 和 (B, Lk, 3) 与 (H, 3, 3) 组合成 (B, H, Lq, Lk)
            # 为了易读与稳定，采用head循环（H通常较小，开销可忽略）
            gates = []
            for h in range(self.num_heads):
                # (B, Lq, 3) @ (3, 3) -> (B, Lq, 3)
                qW = torch.matmul(q_onehot, gate_T[h])  # (B, Lq, 3)
                # (B, Lq, 3) @ (B, 3, Lk) -> (B, Lq, Lk)
                g_h = torch.matmul(qW, k_onehot.transpose(1, 2))  # (B, Lq, Lk)
                gates.append(g_h)
            gates = torch.stack(gates, dim=1)  # (B, H, Lq, Lk)
            
            # 将门控缩放应用到时间偏置；mask会在后续继续生效
            time_bias = time_bias * gates

        # 🎯 修复：仅对非PAD且时间戳有效的位置生效
        # 注意：token_type现在是seq_action_type (0=PAD, 1=曝光, 2=点击)
        if token_type is not None:
            # 非PAD的token（曝光或点击都算有效token）
            valid_token_mask = (token_type >= 1)  # [B, L]
            pair_valid_mask = (valid_token_mask.unsqueeze(2) & valid_token_mask.unsqueeze(1))  # [B, L, L]
        else:
            # 如果没有action_type信息，默认所有位置都可以应用时间偏置
            pair_valid_mask = torch.ones_like(time_delta_hours, dtype=torch.bool)

        if ts_valid_mask is not None:
            pair_ts_valid = (ts_valid_mask.unsqueeze(2) & ts_valid_mask.unsqueeze(1))  # [B, L, L]
        else:
            pair_ts_valid = torch.ones_like(time_delta_hours, dtype=torch.bool)

        pair_mask = (pair_valid_mask & pair_ts_valid).unsqueeze(1)  # [B, 1, L, L]
        time_bias = time_bias.masked_fill(~pair_mask, 0.0)
        
        return time_bias
       

class PointWiseFeedForward(torch.nn.Module):
    '''
    现代化的MLP前馈网络，替换Conv1d以提升表达能力
    🎯 优化：将Dropout置于FFN内部，作用在第一个Linear和激活函数之后
    🔄 支持激活检查点，节省显存
    '''
    def __init__(self, hidden_units, FF_dropout_rate=0.05, use_checkpoint=False):
        super(PointWiseFeedForward, self).__init__()
        
        # 扩展维度提升表达能力（标准做法：4倍扩展）
        inner_hidden = hidden_units * 4
        self.use_checkpoint = use_checkpoint
        
        # 🎯 优化的两层MLP：Dropout在内部，作用在扩展后的隐层上
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_units, inner_hidden),
            torch.nn.SiLU(),  # Swish激活，在推荐系统中效果更好
            torch.nn.Dropout(FF_dropout_rate), # 作用在扩展后的隐层上，更有效的正则化
            torch.nn.Linear(inner_hidden, hidden_units),
            torch.nn.Dropout(FF_dropout_rate), # 输出层也加Dropout
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """改进的权重初始化"""
        for module in self.mlp.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def forward(self, normed_inputs):
        """
        输入: [batch_size, seq_len, hidden_units]
        输出: [batch_size, seq_len, hidden_units]
        """
        # 🔄 FFN 核心计算函数
        def ffn_core(x):
            return self.mlp(x)
        
        # 🔄 使用激活检查点（非重入模式，兼容 autocast）
        if self.training and self.use_checkpoint:
            return checkpoint(ffn_core, normed_inputs, use_reentrant=False)
        else:
            return ffn_core(normed_inputs)


class EnhancedDNN(torch.nn.Module):
    """
    一个带有残差连接和可配置性的增强MLP模块。
    用于将高维拼接特征投影到低维表示。
    MLP_dropout_rate 从0.3降低到0.15，更适合检索/InfoNCE任务
    """
    def __init__(self, input_dim, output_dim, MLP_dropout_rate=0.15, expansion_factor=4):
        super().__init__()
        # 确保隐层维度至少为128，提供足够的表达能力
        hidden_dim = max(output_dim * expansion_factor, 128)

        # 主变换路径 (MLP)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.SiLU(),  # 使用SiLU激活函数
            torch.nn.Dropout(MLP_dropout_rate),
            torch.nn.Linear(hidden_dim, output_dim),
        )

        # 残差连接路径 (如果输入输出维度不同，则需要一个线性投影)
        if input_dim != output_dim:
            self.residual_projection = torch.nn.Linear(input_dim, output_dim)
        else:
            self.residual_projection = torch.nn.Identity()

        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                # 🔧 修正：使用标准的Kaiming He初始化，适用于SiLU激活函数，去掉无效的a参数
                torch.nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        # 计算主路径和残差路径
        mlp_out = self.mlp(x)
        residual_out = self.residual_projection(x)
        return mlp_out + residual_out

class BaselineModel(torch.nn.Module):
    """
    Args:
        user_num: 用户数量
        item_num: 物品数量
        feat_statistics: 特征统计信息，key为特征ID，value为特征数量
        feat_types: 各个特征的特征类型，key为特征类型名称，value为包含的特征ID列表，包括user和item的sparse, array, emb, continual类型
        args: 全局参数

    Attributes:
        user_num: 用户数量
        item_num: 物品数量
        dev: 设备
        norm_first: 是否先归一化
        maxlen: 序列最大长度
        item_emb: Item Embedding Table
        user_emb: User Embedding Table
        sparse_emb: 稀疏特征Embedding Table
        emb_transform: 多模态特征的线性变换
        userdnn: 用户特征拼接后经过的全连接层
        itemdnn: 物品特征拼接后经过的全连接层
    """

    def __init__(self, user_num, item_num, feat_statistics, feat_types, args):  
        super(BaselineModel, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.hidden_units = args.hidden_units  
        self.norm_first = args.norm_first
        self.maxlen = args.maxlen # 101
        self.args = args  # 保存args引用，供其他方法使用
        self.temperature = getattr(args, 'temperature', 0.04)  
        self.enable_rqvae = getattr(args, 'enable_rqvae', True)  
        self.transformer_dropout = getattr(args, 'transformer_dropout', 0.05)
        
        # === 🎯 In-batch 负样本配置（方案B3：所有其他正样本）===
        self.enable_inbatch_negatives = getattr(args, 'enable_inbatch_negatives', False)
        
        if self.enable_inbatch_negatives:
            print(f"🧪 In-batch负样本已启用: 使用所有其他位置的正样本作为额外负样本（自对比掩码保护）")
        
        # 数据监控配置
        self.enable_data_monitoring = getattr(args, 'enable_data_monitoring', False)
        self.monitoring_interval = getattr(args, 'monitoring_interval', 1000)  # 每500步打印一次
        self.monitoring_samples = getattr(args, 'monitoring_samples', 3)  # 每次显示3个样本
        self.training_step_counter = 0
        self.inference_batch_counter = 0
        if self.enable_data_monitoring:
            print(f"🔍 数据监控已启用: 训练每{self.monitoring_interval}步显示{self.monitoring_samples}个样本")

        time_config = get_time_interval_config()
        self.enable_time_features = getattr(args, 'enable_time_features', False)
        self.enable_rope = getattr(args, 'enable_rope', False)
        self.enable_time_bias = getattr(args, 'enable_time_bias', False)
        
        # 🔄 扩展checkpoint配置：支持投影层和连续特征投影
        self.enable_projection_checkpoint = getattr(args, 'enable_projection_checkpoint', True)
        self.enable_continual_projection_checkpoint = getattr(args, 'enable_continual_projection_checkpoint', True)
        if self.enable_projection_checkpoint:
            print("🔄 统一投影层checkpoint已启用")
        if self.enable_continual_projection_checkpoint:
            print("🔄 连续特征投影层checkpoint已启用")
        
        # 📏 Embedding维度配置 - 解耦ID embedding和hidden_units
        embedding_config = get_embedding_config(args)
        # 
        # Field-wise投影功能已移除
        # 🎯 RQ-VAE模式配置
        self.use_precomputed_semantic_ids = getattr(args, 'use_precomputed_semantic_ids', False)
        self.rqvae_models = {}
        
        # 🔥 改进：预计算模式下如果需要复用codebook权重，也要加载RQVAE模型
        if self.enable_rqvae:
            semantic_config = get_semantic_id_config()
            global_config = semantic_config.get('rqvae_alignment', {})
            need_load_rqvae = False
            
            if not self.use_precomputed_semantic_ids:
                # 端到端模式：必须加载RQVAE模型
                need_load_rqvae = True
                print("🎯 端到端模式：加载RQ-VAE模型")
            elif self.use_precomputed_semantic_ids and global_config.get('enable_precompute_codebook_reuse', True):
                # 预计算模式：如果需要复用codebook权重，也要加载RQVAE模型
                need_load_rqvae = True
                print("🎯 预计算模式：为复用codebook权重加载RQ-VAE模型")
            else:
                print("🎯 预计算模式：跳过RQ-VAE模型加载（不复用codebook权重）")
                
            if need_load_rqvae:
                self._load_rqvae_models(args)
        else:
            print("📊 传统模式：使用原始多模态特征")

        # 📏 使用自适应维度的ID embedding，后续通过投影层对齐到hidden_units
        item_id_emb_dim = self._get_adaptive_embedding_dim('item_id', self.item_num)
        user_id_emb_dim = self._get_adaptive_embedding_dim('user_id', self.user_num)
        self.item_emb = torch.nn.Embedding(self.item_num + 1, item_id_emb_dim, padding_idx=0)
        self.user_emb = torch.nn.Embedding(self.user_num + 1, user_id_emb_dim, padding_idx=0)
        print(f"📏 ID embedding维度: item_id={item_id_emb_dim}, user_id={user_id_emb_dim}")
         
        # 🎯 HSTU改进：智能位置编码策略
        self.use_absolute_pos_emb = (not self.enable_rope and 
                                   not getattr(args, 'enable_relative_bias', False))
        if self.use_absolute_pos_emb:
            self.pos_emb = torch.nn.Embedding(2 * args.maxlen + 1, args.hidden_units, padding_idx=0)
            print("📍 使用传统绝对位置编码")
        else:
            self.pos_emb = None
            if self.enable_rope:
                print("🎯 使用RoPE旋转位置编码")
            if getattr(args, 'enable_relative_bias', False):
                print("🎯 使用HSTU相对位置偏置(RAB)")
        
        # 🎯 HSTU模式配置检查
        attention_mode = getattr(args, 'attention_mode', 'sdpa')
        if attention_mode == 'hstu':
            print(f"🎯 HSTU注意力模式已启用: "
                  f"RoPE={self.enable_rope}, "
                  f"时间偏置={self.enable_time_bias}, "
                  f"相对偏置={getattr(args, 'enable_relative_bias', False)}")
            print(f"🎯 HSTU配置: 非负约束=True, Q/K归一化=RMSNorm, Dropout位置=权重上")

        self.emb_dropout = torch.nn.Dropout(p=args.emb_dropout_rate)
        self.emb_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-5)
        
        # 🔥 新增：ID-dropout 配置 - 防止过度依赖ID统计信号，迫使模型学习内容特征
        self.drop_id_rate = getattr(args, 'id_dropout_rate', 0.05)  # 默认5%概率丢弃ID
        self.id_dropout_mode = getattr(args, 'id_dropout_mode', 'sequence')  # 'sequence' or 'token'
        
        # === [新增] 动作类型 margin 配置（仅训练期在InfoNCE中生效） ===
        self.enable_action_margin = getattr(args, 'enable_action_margin', False)
        # 0=曝光, 1=点击；建议 γ(click)=0.0, γ(exposure)=0.2~0.5 可做网格搜索
        self.action_margin_click = getattr(args, 'action_margin_click', 0.0)
        self.action_margin_exposure = getattr(args, 'action_margin_exposure', 0.2)
        
        # Gate融合功能已移除，使用简单拼接+统一投影
        
        if self.drop_id_rate > 0:
            print(f"🔥 ID-dropout已启用: rate={self.drop_id_rate}, mode={self.id_dropout_mode}, 增强冷启动泛化能力")
        
        if self.enable_action_margin:
            print(f"🎯 Action-aware margin已启用: γ(click)={self.action_margin_click}, γ(exposure)={self.action_margin_exposure}")

        self.sparse_emb = torch.nn.ModuleDict() # 一个用来存放所有稀疏特征（如品牌、类别等）的嵌入层（Embedding Layers）的容器。 
        self.emb_transform = torch.nn.ModuleDict() # 传统多模态

        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self._init_feat_info(feat_statistics, feat_types)

        # 🎯 RQ-VAE模式配置：区分预计算和端到端模式
        if self.enable_rqvae and self.use_precomputed_semantic_ids:
            print(f"🎯 预计算模式：跳过多模态特征处理，使用预计算的semantic_id")
            self.original_emb_feat = {}  # 预计算模式不需要原始特征
        elif self.enable_rqvae:
            print(f"🎯 端到端RQ-VAE模式：原始多模态特征 {list(self.ITEM_EMB_FEAT.keys())} 将实时转换为semantic id")
            self.original_emb_feat = self.ITEM_EMB_FEAT.copy()
            self.ITEM_EMB_FEAT = {}
            print(f"✅ 端到端RQVAE模式：原始特征已转换，ITEM_EMB_FEAT已清空，semantic特征由Dataset管理")
        else:
            print(f"📊 传统模式：使用原始多模态特征 {list(self.ITEM_EMB_FEAT.keys())}")
            self.original_emb_feat = {}

        # 用户维度计算
        userdim = 0
        userdim += user_id_emb_dim
        # 用户稀疏特征
        for k in self.USER_SPARSE_FEAT:
            dim = self._get_adaptive_embedding_dim(k, self.USER_SPARSE_FEAT[k])
            if dim is not None:
                userdim += dim
        # 用户数组特征
        for k in self.USER_ARRAY_FEAT:
            dim = self._get_adaptive_embedding_dim(k, self.USER_ARRAY_FEAT[k])
            if dim is not None:
                userdim += dim
        # 用户连续特征
        if len(self.USER_CONTINUAL_FEAT) > 0:
            embedding_config = get_embedding_config(self.args)
            continual_config = embedding_config.get('continual_features_config', {})
            user_continual_proj_dim = continual_config.get('user_continual_proj_dim', 16)
            userdim += user_continual_proj_dim  # 使用投影后的维度
        
        # 基础item维度计算（不包含时间和语义ID）
        base_itemdim = 0
        # 物品ID
        base_itemdim += item_id_emb_dim
        # 物品稀疏特征
        for k in self.ITEM_SPARSE_FEAT:
            dim = self._get_adaptive_embedding_dim(k, self.ITEM_SPARSE_FEAT[k])
            if dim is not None:
                base_itemdim += dim
        # 物品数组特征
        for k in self.ITEM_ARRAY_FEAT:
            dim = self._get_adaptive_embedding_dim(k, self.ITEM_ARRAY_FEAT[k])
            if dim is not None:
                base_itemdim += dim
        # 物品连续特征
        if len(self.ITEM_CONTINUAL_FEAT) > 0:
            embedding_config = get_embedding_config(self.args)
            continual_config = embedding_config.get('continual_features_config', {})
            item_continual_proj_dim = continual_config.get('item_continual_proj_dim', 24)
            base_itemdim += item_continual_proj_dim  # 使用投影后的维度
        
        # 🎯 传统多模态特征维度计算（仅在非RQ-VAE模式下）
        if not self.enable_rqvae:
            for k in self.ITEM_EMB_FEAT:
                # 传统多模态特征经过线性变换后维度为hidden_units
                base_itemdim += args.hidden_units
                print(f"📏 传统多模态特征 {k}: 线性变换后维度 {args.hidden_units}")
        
        # 🔧 时间特征维度单独计算
        time_dim = 0
        if self.enable_time_features:
            embedding_config = get_embedding_config(self.args)
            time_config = embedding_config.get('time_features_config', {})
            time_gap_dim = time_config.get('time_gap_embedding_dim', 16)
            action_type_dim = time_config.get('action_type_embedding_dim', 8)
            # 按特征分别累加，避免将action_type错误当成time_gap_dim
            sparse_time_detail = []
            for _fid in self.SEQ_TIME_SPARSE_FEAT:
                if _fid == 'action_type':
                    time_dim += action_type_dim
                    sparse_time_detail.append(f"{_fid}:{action_type_dim}")
                else:
                    time_dim += time_gap_dim
                    sparse_time_detail.append(f"{_fid}:{time_gap_dim}")
            if len(self.SEQ_TIME_CONTINUAL_FEAT) > 0:
                absolute_time_proj_dim = time_config.get('absolute_time_proj_dim', 32)
                time_dim += absolute_time_proj_dim
            else:
                absolute_time_proj_dim = 0
            print(f"🕐 时间特征维度详细计算: sparse([{', '.join(sparse_time_detail)}]) + continual={absolute_time_proj_dim} => time_dim={time_dim}")
        
        semantic_id_dim = 0
        if self.enable_rqvae:
            semantic_config = get_semantic_id_config()
            active_features = [fid for fid in getattr(args, 'mm_emb_id', ['81']) if fid in semantic_config['semantic_id_features']]
            for feature_id in active_features:
                feature_config = semantic_config['semantic_id_features'][feature_id]
                single_emb_dim = feature_config['embedding_dim']
                num_codebooks = feature_config['array_length']
                fusion_mode = feature_config.get('fusion_mode', 'sum')
                
                # 🎯 改进：根据融合模式计算维度
                if fusion_mode in ['sum', 'weighted_sum', 'hybrid']:
                    # 求和/加权求和/混合模式：输出维度 = 单个embedding维度
                    feature_dim = single_emb_dim
                    print(f"🔧 语义ID特征{feature_id}: {fusion_mode}模式, 输出维度={feature_dim}")
                elif fusion_mode == 'concat':
                    # 拼接模式：输出维度 = embedding_dim * num_codebooks
                    feature_dim = single_emb_dim * num_codebooks
                    print(f"🔧 语义ID特征{feature_id}: {fusion_mode}模式, {single_emb_dim}*{num_codebooks}={feature_dim}维")
                else:
                    # 未知模式，fallback到拼接
                    feature_dim = single_emb_dim * num_codebooks
                    print(f"⚠️ 语义ID特征{feature_id}: 未知融合模式{fusion_mode}, fallback到concat: {feature_dim}维")
                
                semantic_id_dim += feature_dim
        
        # 序列维度：包含时间特征和语义ID
        itemdim_seq = base_itemdim + time_dim + semantic_id_dim
        
        # 候选item维度：包含语义ID但不含时间特征
        itemdim_cand = base_itemdim + semantic_id_dim

        print(f"🧮 维度核算明细: userdim={userdim}, base_itemdim={base_itemdim}, time_dim={time_dim}, semantic_id_dim={semantic_id_dim}")
        print(f"📊 模型维度信息：userdim={userdim}, itemdim_seq={itemdim_seq}, itemdim_cand={itemdim_cand}")
        print(f"📊 特征统计：USER_SPARSE={len(self.USER_SPARSE_FEAT)}, USER_CONTINUAL={len(self.USER_CONTINUAL_FEAT)}, ITEM_SPARSE={len(self.ITEM_SPARSE_FEAT)}, ITEM_CONTINUAL={len(self.ITEM_CONTINUAL_FEAT)}, ITEM_EMB={len(self.ITEM_EMB_FEAT)}")
        print(f"📊 时间特征维度：{time_dim} (启用: {self.enable_time_features}, SEQ_TIME_SPARSE={len(self.SEQ_TIME_SPARSE_FEAT)}, SEQ_TIME_CONTINUAL={len(self.SEQ_TIME_CONTINUAL_FEAT)})")
        
        # 🎯 延迟初始化统一投影层：在field projection构建后进行
        self._deferred_unified_projection_init = {
            'userdim': userdim,
            'itemdim_seq': itemdim_seq, 
            'itemdim_cand': itemdim_cand,
            'args': args
        }
        
        # 直接初始化统一投影层
        if hasattr(self, '_deferred_unified_projection_init'):
                deferred_init = self._deferred_unified_projection_init
                args = deferred_init['args']
                
                # 🎯 候选侧头部类型开关
                item_cand_head = getattr(args, 'item_cand_head', 'linear')  # 保持当前默认行为
                
                self.unified_item_seq_projection = create_candidate_head(
                    deferred_init['itemdim_seq'], 
                    args.hidden_units, 
                    head_type=item_cand_head,
                    dropout_rate=args.MLP_dropout_rate
                )
                self.unified_user_projection = create_candidate_head(
                    deferred_init['userdim'], 
                    args.hidden_units, 
                    head_type=item_cand_head,
                    dropout_rate=args.MLP_dropout_rate
                )
                # 候选侧：可切换头部类型
                self.unified_item_cand_projection = create_candidate_head(
                    deferred_init['itemdim_cand'], 
                    args.hidden_units, 
                    head_type=item_cand_head,
                    dropout_rate=args.MLP_dropout_rate
                )
                
                print(f"🔄 传统模式统一投影层已初始化: user({deferred_init['userdim']}->{args.hidden_units}), item_seq({deferred_init['itemdim_seq']}->{args.hidden_units}), item_cand({deferred_init['itemdim_cand']}->{args.hidden_units}, head={item_cand_head})")
                
                # Gate融合网络层初始化已移除
                
                delattr(self, '_deferred_unified_projection_init')
        
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-5)
        
        for i in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-5)
            self.attention_layernorms.append(new_attn_layernorm)
             
            new_attn_layer = FlashMultiHeadAttention(
                args.hidden_units, 
                args.num_heads, 
                args.transformer_dropout,
                enable_rope=self.enable_rope,
                enable_time_features=self.enable_time_features,
                enable_time_bias=self.enable_time_bias,
                attention_mode=getattr(args, 'attention_mode', 'hstu'),
                enable_relative_bias=False,
                use_checkpoint=True
            )  
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-5)
            self.forward_layernorms.append(new_fwd_layernorm)

            # 🎯 将transformer_dropout传递给FFN，实现内部正则化 + 激活检查点
            new_fwd_layer = PointWiseFeedForward(
                args.hidden_units, 
                FF_dropout_rate=self.transformer_dropout,
                use_checkpoint=True
            )
            self.forward_layers.append(new_fwd_layer)
            

        embedding_config = get_embedding_config(self.args)
        
        # 🎯 自适应embedding维度分配 - 用户稀疏特征
        for k in self.USER_SPARSE_FEAT:
            emb_dim = self._get_adaptive_embedding_dim(k, self.USER_SPARSE_FEAT[k])
            if emb_dim is None: 
                print(f"🚫 用户稀疏特征 {k} 已跳过（配置为移除）") 
                continue
            self.sparse_emb[k] = torch.nn.Embedding(self.USER_SPARSE_FEAT[k] + 1, emb_dim, padding_idx=0)
            print(f"📏 用户稀疏特征 {k}: vocab_size={self.USER_SPARSE_FEAT[k]}, emb_dim={emb_dim}")
            
        # 🎯 自适应embedding维度分配 - 物品稀疏特征
        for k in self.ITEM_SPARSE_FEAT:
            emb_dim = self._get_adaptive_embedding_dim(k, self.ITEM_SPARSE_FEAT[k])
            if emb_dim is None:  # 特征被配置为移除
                print(f"🚫 物品稀疏特征 {k} 已跳过（配置为移除）")
                continue
                
            self.sparse_emb[k] = torch.nn.Embedding(self.ITEM_SPARSE_FEAT[k] + 1, emb_dim, padding_idx=0)
            print(f"📏 物品稀疏特征 {k}: vocab_size={self.ITEM_SPARSE_FEAT[k]}, emb_dim={emb_dim}")
            
        # 🎯 自适应embedding维度分配 - 物品数组特征
        for k in self.ITEM_ARRAY_FEAT:
            emb_dim = self._get_adaptive_embedding_dim(k, self.ITEM_ARRAY_FEAT[k])
            if emb_dim is None:  # 特征被配置为移除
                print(f"🚫 物品数组特征 {k} 已跳过（配置为移除）")
                continue
                
            self.sparse_emb[k] = torch.nn.Embedding(self.ITEM_ARRAY_FEAT[k] + 1, emb_dim, padding_idx=0)
            print(f"📏 物品数组特征 {k}: vocab_size={self.ITEM_ARRAY_FEAT[k]}, emb_dim={emb_dim}")
            
        # 🎯 自适应embedding维度分配 - 用户数组特征  
        for k in self.USER_ARRAY_FEAT:
            emb_dim = self._get_adaptive_embedding_dim(k, self.USER_ARRAY_FEAT[k])
            if emb_dim is None:  # 特征被配置为移除
                print(f"🚫 用户数组特征 {k} 已跳过（配置为移除）")
                continue
                
            self.sparse_emb[k] = torch.nn.Embedding(self.USER_ARRAY_FEAT[k] + 1, emb_dim, padding_idx=0)
            print(f"📏 用户数组特征 {k}: vocab_size={self.USER_ARRAY_FEAT[k]}, emb_dim={emb_dim}")

        # 稀疏时间特征
        if self.enable_time_features:
            for feat_id in self.SEQ_TIME_SPARSE_FEAT:
                vocab_size = self.SEQ_TIME_SPARSE_FEAT[feat_id]
                emb_dim = self._get_adaptive_embedding_dim(f'{feat_id}', vocab_size)
                if emb_dim is None:  # 特征被配置为移除
                    print(f"🚫 时间稀疏特征 {feat_id} 已跳过（配置为移除）")
                    continue          
                self.sparse_emb[feat_id] = torch.nn.Embedding(vocab_size + 1, emb_dim, padding_idx=0)
                print(f"🕐 时间稀疏特征embedding已初始化: {feat_id}, vocab_size={vocab_size}, emb_dim={emb_dim}")
        
        # 🕐 绝对时间特征投影层
        if self.enable_time_features:
            time_config = embedding_config.get('time_features_config', {})
            absolute_time_proj_dim = time_config.get('absolute_time_proj_dim', 32)  # 从256降至32
            
            # 🎯 动态设置输入维度为所有连续时间特征的数量
            num_continual_time_feats = len(self.SEQ_TIME_CONTINUAL_FEAT)
            if num_continual_time_feats > 0:
                self.absolute_time_projection = torch.nn.Linear(num_continual_time_feats, absolute_time_proj_dim)
                print(f"🕐 连续时间特征投影层已初始化: {num_continual_time_feats} -> {absolute_time_proj_dim}")
            else:
                self.absolute_time_projection = None
                print("🕐 未检测到连续时间特征，跳过投影层初始化")
        
        # 🎯 用户和物品连续特征投影层
        continual_config = embedding_config.get('continual_features_config', {})
        
        # 用户连续特征投影层
        num_user_continual_feats = len(self.USER_CONTINUAL_FEAT)
        if num_user_continual_feats > 0:
            user_continual_proj_dim = continual_config.get('user_continual_proj_dim', 16)
            self.user_continual_projection = torch.nn.Linear(num_user_continual_feats, user_continual_proj_dim)
            print(f"👤 用户连续特征投影层已初始化: {num_user_continual_feats} -> {user_continual_proj_dim}")
        else:
            self.user_continual_projection = None
            print("👤 未检测到用户连续特征，跳过投影层初始化")
        
        # 物品连续特征投影层
        num_item_continual_feats = len(self.ITEM_CONTINUAL_FEAT)
        if num_item_continual_feats > 0:
            item_continual_proj_dim = continual_config.get('item_continual_proj_dim', 24)
            self.item_continual_projection = torch.nn.Linear(num_item_continual_feats, item_continual_proj_dim)
            print(f"🎯 物品连续特征投影层已初始化: {num_item_continual_feats} -> {item_continual_proj_dim}")
        else:
            self.item_continual_projection = None
            print("🎯 未检测到物品连续特征，跳过投影层初始化")
        
        # 🎯 连续特征信息输出（补充缺失的日志）
        if len(self.USER_CONTINUAL_FEAT) > 0:
            print(f"📏 用户连续特征: 数量={len(self.USER_CONTINUAL_FEAT)}, 特征列表={list(self.USER_CONTINUAL_FEAT)}")
        if len(self.ITEM_CONTINUAL_FEAT) > 0:
            print(f"📏 物品连续特征: 数量={len(self.ITEM_CONTINUAL_FEAT)}, 特征列表={list(self.ITEM_CONTINUAL_FEAT)}")
        if self.enable_time_features and len(self.SEQ_TIME_CONTINUAL_FEAT) > 0:
            print(f"📏 时间连续特征: 数量={len(self.SEQ_TIME_CONTINUAL_FEAT)}, 特征列表={list(self.SEQ_TIME_CONTINUAL_FEAT)}")
        
        # 🎯 改进的semantic_id特征embedding：支持RQ-VAE对齐的处理方式
        if self.enable_rqvae:
            semantic_config = get_semantic_id_config()
            active_features = [fid for fid in getattr(args, 'mm_emb_id', ['81']) if fid in semantic_config['semantic_id_features']]
            self.semantic_embeds = torch.nn.ModuleDict()
            self.semantic_layer_weights = torch.nn.ParameterDict()  # 🎯 新增：可学习的per-layer权重
            self.semantic_fusion_mode = {}  # 记录每个特征的融合模式
            self.semantic_norms = torch.nn.ModuleDict()  # 🎯 新增：融合后归一化层（按特征）
            
            # 🎯 全局配置
            global_config = semantic_config.get('rqvae_alignment', {})
            self.enable_codebook_reuse = global_config.get('enable_codebook_reuse', False)
            self.enable_sid_dropout = global_config.get('enable_sid_dropout', False)
            self.sid_dropout_rate = global_config.get('sid_dropout_rate', 0.05)
            self.enable_post_fusion_norm = global_config.get('enable_post_fusion_norm', False)
            self.post_fusion_norm_type = global_config.get('post_fusion_norm_type', 'layernorm')  # 'layernorm'|'l2'
            self.post_fusion_norm_eps = global_config.get('post_fusion_norm_eps', 1e-5)
            
            for feature_id in active_features:
                feature_config = semantic_config['semantic_id_features'][feature_id]
                feature_name = feature_config['feature_name']  # 如 'semantic_81'
                vocab_size = feature_config['vocab_size']
                embedding_dim = feature_config['embedding_dim']
                num_codebooks = feature_config['array_length']
                padding_value = feature_config.get('padding_value', 0)
                
                # 🎯 读取特征级配置
                fusion_mode = feature_config.get('fusion_mode', 'sum')
                enable_layer_weights = feature_config.get('enable_layer_weights', True)
                reuse_codebook_weights = feature_config.get('reuse_codebook_weights', True)
                
                self.semantic_fusion_mode[feature_name] = fusion_mode

                # 🎯 改进：明确复用特征的适用条件和隔离逻辑
                # 🎯 改进的codebook权重复用逻辑：
                # 复用条件（所有条件必须同时满足）：
                # 1. 全局开关enable_codebook_reuse开启
                # 2. 特征级配置reuse_codebook_weights开启  
                # 3. RQ-VAE模型已加载且包含对应特征
                # 4. 预计算模式需额外检查enable_precompute_codebook_reuse开关
                should_reuse_codebook = (
                    self.enable_codebook_reuse and
                    reuse_codebook_weights and
                    hasattr(self, 'rqvae_models') and
                    feature_id in self.rqvae_models
                )
                
                # 🔥 预计算模式额外检查
                if self.use_precomputed_semantic_ids:
                    global_config = semantic_config.get('rqvae_alignment', {})
                    should_reuse_codebook = should_reuse_codebook and global_config.get('enable_precompute_codebook_reuse', True)
                
                if should_reuse_codebook:
                    # 端到端模式：复用RQ-VAE的码本权重
                    emb_list = []
                    rqvae_model = self.rqvae_models[feature_id]
                    for m in range(num_codebooks):
                        codebook_weight = rqvae_model.quantizer.codebooks[m].weight.data.clone()
                        # 扩展词表以包含特殊值（padding等）
                        if codebook_weight.shape[0] < vocab_size:
                            extra_rows = vocab_size - codebook_weight.shape[0]
                            extra_embeddings = torch.randn(extra_rows, embedding_dim, device=codebook_weight.device) * 0.02
                            codebook_weight = torch.cat([codebook_weight, extra_embeddings], dim=0)
                        
                        emb = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_value)
                        emb.weight.data.copy_(codebook_weight)
                        
                        # 🎯 标记为从码本复用，避免重新初始化
                        emb._initialized_from_codebook = True
                        
                        # 🎯 可选微调：允许对复用权重进行少量调整
                        if feature_config.get('allow_fine_tune', True):
                            emb.weight.requires_grad = True
                        else:
                            emb.weight.requires_grad = False
                            
                        emb_list.append(emb)
                    
                    print(f"🎯 特征{feature_name}: 复用RQ码本权重，允许微调={feature_config.get('allow_fine_tune', True)}")
                else:
                    # 预计算模式或非复用模式：新建embedding表
                    emb_list = []
                    for _m in range(num_codebooks):
                        emb = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_value)
                        # 标记为新建，需要正常初始化
                        emb._initialized_from_codebook = False
                        emb_list.append(emb)
                    
                    reuse_reason = []
                    if not self.enable_codebook_reuse:
                        reuse_reason.append("全局复用开关关闭")
                    if not reuse_codebook_weights:
                        reuse_reason.append("特征级复用配置关闭")
                    if self.use_precomputed_semantic_ids:
                        reuse_reason.append("预计算模式")
                    if not (hasattr(self, 'rqvae_models') and feature_id in self.rqvae_models):
                        reuse_reason.append("RQ模型未加载")
                        
                    print(f"🎯 特征{feature_name}: 新建embedding表 (原因: {', '.join(reuse_reason)})")
                
                self.semantic_embeds[feature_name] = torch.nn.ModuleList(emb_list)

                # 🎯 可学习的per-layer权重（仅用于加权求和模式）
                if enable_layer_weights and fusion_mode == 'weighted_sum':
                    layer_weights = torch.nn.Parameter(torch.ones(num_codebooks))
                    self.semantic_layer_weights[feature_name] = layer_weights
                    print(f"🎯 特征{feature_name}: 启用可学习per-layer权重 (weighted_sum模式)")
                elif fusion_mode == 'sum':
                    print(f"🎯 特征{feature_name}: 使用等权求和 (sum模式，无可学习权重)")
                elif fusion_mode == 'concat':
                    print(f"🎯 特征{feature_name}: 使用拼接模式 (concat模式，无权重融合)")
                elif fusion_mode == 'hybrid':
                    # hybrid模式需要特殊权重
                    hybrid_weight = torch.nn.Parameter(torch.tensor(0.7))  # sum分支权重
                    self.semantic_layer_weights[f"{feature_name}_hybrid"] = hybrid_weight
                    print(f"🎯 特征{feature_name}: 启用混合模式权重 (hybrid模式)")

                # 🎯 根据融合模式创建融合后归一化层（可选）
                if self.enable_post_fusion_norm:
                    single_emb_dim = embedding_dim
                    if fusion_mode in ['sum', 'weighted_sum', 'hybrid']:
                        feature_dim = single_emb_dim
                    elif fusion_mode == 'concat':
                        feature_dim = single_emb_dim * num_codebooks
                    else:
                        feature_dim = single_emb_dim
                    if self.post_fusion_norm_type == 'layernorm':
                        self.semantic_norms[feature_name] = torch.nn.LayerNorm(feature_dim, eps=self.post_fusion_norm_eps).to(self.dev)
                        print(f"🎯 特征{feature_name}: 启用融合后LayerNorm (dim={feature_dim})")
                    elif self.post_fusion_norm_type == 'l2':
                        # L2范数归一化无需可学习参数，这里仅记录以便前向时应用
                        print(f"🎯 特征{feature_name}: 启用融合后L2归一化 (dim={feature_dim})")

                mode_desc = "预计算模式" if self.use_precomputed_semantic_ids else "端到端模式"
                print(f"🎯 {mode_desc} semantic_id已初始化: {feature_name}, "
                      f"codebooks={num_codebooks}, vocab_size={vocab_size}, "
                      f"emb_dim={embedding_dim}, fusion_mode={fusion_mode}")
        
        
        if not self.enable_rqvae:
            # 传统模式：使用线性变换保持原始维度，后续通过统一投影对齐
            for k in self.ITEM_EMB_FEAT:
                self.emb_transform[k] = torch.nn.Linear(self.ITEM_EMB_FEAT[k], args.hidden_units) # emb是因为主办方提供的就是embbedding
                print(f"📏 多模态特征 {k}: 保持原始维度 {self.ITEM_EMB_FEAT[k]}")

        # 🎯 统一权重初始化：避免外部初始化覆盖内部精心设计的初始化
        self._init_model_weights()
    
    
    def _apply_id_dropout(self, *embeddings):
        """
        🔥 应用ID Dropout - 防止过度依赖ID统计信号，增强冷启动泛化能力
        支持两种模式：sequence级别（整条序列）和token级别（单个token）
        
        Args:
            *embeddings: 一个或多个embedding tensor，形状为 [batch_size, seq_len, emb_dim]
            
        Returns:
            若输入单个embedding，返回处理后的单个tensor
            若输入多个embedding，返回处理后的tuple
        """
        if not self.training or self.drop_id_rate <= 0:
            return embeddings[0] if len(embeddings) == 1 else embeddings
        
        processed_embeddings = []
        
        for embedding in embeddings:
            if self.id_dropout_mode == 'sequence':
                # 序列级别dropout：按概率将整条序列的ID embedding置零
                # 创建与batch大小相同的mask，按照概率drop_id_rate置为0
                batch_size = embedding.size(0)
                drop_mask = torch.rand(batch_size, 1, 1, device=embedding.device) > self.drop_id_rate
                dropped_embedding = embedding * drop_mask.float()
            else:
                # token级别dropout：按概率将单个token的ID embedding置零
                # 更细粒度的控制，但可能影响序列的连贯性
                drop_mask = torch.rand_like(embedding[:, :, :1]) > self.drop_id_rate
                dropped_embedding = embedding * drop_mask.float()
            
            processed_embeddings.append(dropped_embedding)
        
        # 返回格式与输入保持一致
        return processed_embeddings[0] if len(embeddings) == 1 else tuple(processed_embeddings)

    def _load_rqvae_models(self, args):
        """
        加载预训练的RQVAE模型用于端到端推理
        🚀 支持多特征(81、82等)动态加载
        """
        import os
        from pathlib import Path
        from model_rqvae import RQVAE
        
        print("🎯 端到端模式：加载预训练RQVAE模型...")
        
        # 获取RQVAE配置
        config = get_rqvae_config()
        
        # 搜索RQVAE模型文件路径
        search_paths = []
        cache_dir = os.environ.get('USER_CACHE_PATH')
        
        if cache_dir:
            search_paths.append(Path(cache_dir))
        else:
            raise ValueError("USER_CACHE_PATH环境变量未设置，无法加载RQVAE模型")
        
        # 🚀 动态支持多特征：根据mm_emb_id决定要加载的特征
        active_features = [fid for fid in getattr(args, 'mm_emb_id', ['81','82']) if fid in config]
        print(f"🎯 将加载以下特征的RQVAE模型: {active_features}")
        
        for feature_id in active_features:
            model_loaded = False
            rqvae_config = config[feature_id]
            
            # 🎯 创建新的RQVAE模型实例
            rqvae_model = RQVAE(
                input_dim=rqvae_config["input_dim"],
                hidden_channels=rqvae_config["hidden_channels"],
                latent_dim=rqvae_config["latent_dim"],
                num_codebooks=rqvae_config["num_codebooks"],
                codebook_size=rqvae_config["codebook_size"],
                ema_decay=rqvae_config.get("ema_decay", 0.99),
                commitment_cost=rqvae_config.get("commitment_cost", 0.25),
                diversity_gamma=rqvae_config.get("diversity_gamma", 0.1),
                encoder_dropout=rqvae_config.get("encoder_dropout", 0.1),
                decoder_dropout=rqvae_config.get("decoder_dropout", 0.1),
                device=args.device
            ).to(args.device)
            # 🎯 新RQVAE模型使用EMA更新机制，不需要手动设置初始化状态
            # 搜索模型文件 - 按step数从大到小排序
            best_model_path = None
            best_step = -1
            
            for search_path in search_paths:
                # 搜索符合新模式的模型文件
                pattern_paths = list(search_path.glob(f"global_step*.rqvae_feat_{feature_id}_final*/model.pt"))
                
                for model_path in pattern_paths:
                    if model_path.exists():
                        # 从路径中提取step数
                        path_parts = str(model_path).split(os.sep)
                        for part in path_parts:
                            if part.startswith(f"global_step") and f"rqvae_feat_{feature_id}_final" in part:
                                try:
                                    # 提取step数字：global_step{number}.rqvae_feat_{feature_id}_final
                                    step_str = part.split('.')[0].replace('global_step', '')
                                    step_num = int(step_str)
                                    
                                    # 如果找到更大的step数，更新最佳模型
                                    if step_num > best_step:
                                        best_step = step_num
                                        best_model_path = model_path
                                        print(f"🔍 发现step {step_num}的模型: {model_path}")
                                except ValueError:
                                    continue
                                break
            
            # 加载最佳模型(step数最大的)
            if best_model_path:
                try:
                    print(f"✅ 从 {best_model_path} 加载特征 {feature_id} 的RQVAE模型 (step: {best_step})...")
                    state_dict = torch.load(best_model_path, map_location=args.device)
                    # 使用软性加载，允许键名不匹配
                    missing_keys, unexpected_keys = rqvae_model.load_state_dict(state_dict, strict=False)
                    
                    # 记录加载状态
                    if missing_keys:
                        print(f"⚠️ 缺失的键: {missing_keys}")
                    if unexpected_keys:
                        print(f"⚠️ 意外的键: {unexpected_keys}")
                    
                    rqvae_model.eval()  # 设为推理模式
                    
                    # 冻结所有参数
                    for param in rqvae_model.parameters():
                        param.requires_grad = False
                    
                    self.rqvae_models[feature_id] = rqvae_model
                    print(f"✅ 特征 {feature_id} 的RQVAE模型已冻结并加载完成 (最终step: {best_step})")
                    model_loaded = True
                except Exception as e:
                    print(f"❌ 从 {best_model_path} 加载失败: {e}")
                    model_loaded = False
            
            if not model_loaded:
                print(f"❌ 未找到特征 {feature_id} 的RQVAE模型文件")
                print("请确保已运行RQ-VAE预训练或提供正确的模型路径")
                raise FileNotFoundError(f"RQVAE model for feature {feature_id} not found")
        
        print(f"🎯 端到端RQVAE：已加载 {len(self.rqvae_models)} 个特征的模型")
    
    def _init_model_weights(self):
        """
        统一的模型权重初始化方法
        避免外部初始化覆盖内部精心设计的初始化策略
        """
        # 1. 📏 ID embedding初始化：使用更合理的标准差
        init_std = 0.02  # 增加标准差，参考BERT/GPT等模型的embedding初始化
        torch.nn.init.normal_(self.item_emb.weight.data, mean=0.0, std=init_std)
        torch.nn.init.normal_(self.user_emb.weight.data, mean=0.0, std=init_std)
        
        # 确保padding_idx=0的embedding仍然是严格的零
        self.item_emb.weight.data[0, :] = 0
        self.user_emb.weight.data[0, :] = 0
        
        # 2. 位置编码使用较小的初始化范围（如果启用传统位置编码）
        if self.pos_emb is not None:  # 🎯 条件性初始化位置编码
            init_range = 0.02  # 更小的初始化范围，参考BERT
            torch.nn.init.normal_(self.pos_emb.weight.data, mean=0.0, std=init_range)
            self.pos_emb.weight.data[0, :] = 0  # padding_idx = 0
        
        # 3. 📏 初始化小维度稀疏特征Embedding（无投影层）
        init_range = 0.02
        for emb in self.sparse_emb.values():
            torch.nn.init.normal_(emb.weight.data, mean=0.0, std=init_range)
            emb.weight.data[0, :] = 0  # padding_idx = 0
        print(f"📏 自适应稀疏特征embedding正常初始化：std={init_range}")
        
        # 4. 初始化多模态特征变换层（如果存在）- 这些层需要正常初始化以处理多模态信息
        for transform in self.emb_transform.values():
            torch.nn.init.xavier_normal_(transform.weight.data)
            if transform.bias is not None:
                torch.nn.init.zeros_(transform.bias.data)
        
        # 5. 初始化LayerNorm层 
        for ln in self.attention_layernorms + self.forward_layernorms + [self.last_layernorm, self.emb_layernorm]:
            torch.nn.init.ones_(ln.weight.data)
            torch.nn.init.zeros_(ln.bias.data)
        
        # 6. 初始化绝对时间特征投影层（如果启用时间特征且投影层存在）
        if self.enable_time_features and hasattr(self, 'absolute_time_projection') and self.absolute_time_projection is not None:
            torch.nn.init.xavier_normal_(self.absolute_time_projection.weight.data)
            if self.absolute_time_projection.bias is not None:
                torch.nn.init.zeros_(self.absolute_time_projection.bias.data)
            print("🕐 绝对时间特征投影层已初始化")
        
        # 7. 初始化用户连续特征投影层
        if hasattr(self, 'user_continual_projection') and self.user_continual_projection is not None:
            torch.nn.init.xavier_normal_(self.user_continual_projection.weight.data)
            if self.user_continual_projection.bias is not None:
                torch.nn.init.zeros_(self.user_continual_projection.bias.data)
            print("👤 用户连续特征投影层已初始化")
        
        # 8. 初始化物品连续特征投影层
        if hasattr(self, 'item_continual_projection') and self.item_continual_projection is not None:
            torch.nn.init.xavier_normal_(self.item_continual_projection.weight.data)
            if self.item_continual_projection.bias is not None:
                torch.nn.init.zeros_(self.item_continual_projection.bias.data)
            print("🎯 物品连续特征投影层已初始化")

        # 9. 🎯 改进的semantic embedding参数初始化（如果启用RQ-VAE）
        if self.enable_rqvae and hasattr(self, 'semantic_embeds'):
            semantic_config = get_semantic_id_config()
            global_config = semantic_config.get('rqvae_alignment', {})
            
            for feature_name in self.semantic_embeds.keys():
                # 🎯 智能初始化：复用码本权重的embedding无需重新初始化
                for emb_table in self.semantic_embeds[feature_name]:
                    if emb_table.weight.requires_grad:
                        # 只对需要训练的embedding表进行初始化
                        if not hasattr(emb_table, '_initialized_from_codebook'):
                            # 未从RQ码本复用的embedding表使用正常初始化
                            init_range = 0.02
                            torch.nn.init.normal_(emb_table.weight.data, mean=0.0, std=init_range)
                        # padding_idx行已由PyTorch自动处理
                
                # 🎯 初始化可学习的per-layer权重
                if (hasattr(self, 'semantic_layer_weights') and 
                    feature_name in self.semantic_layer_weights):
                    # 初始化为接近均匀的权重，避免某层过度支配
                    layer_weights = self.semantic_layer_weights[feature_name]
                    torch.nn.init.constant_(layer_weights.data, 1.0)  # 等权初始化
                    print(f"🎯 {feature_name}: per-layer权重已等权初始化")
            
            # 🎯 初始化混合融合的投影层（如果存在）
            for attr_name in dir(self):
                if attr_name.startswith('_hybrid_proj_'):
                    proj_layer = getattr(self, attr_name)
                    if isinstance(proj_layer, torch.nn.Linear):
                        torch.nn.init.xavier_normal_(proj_layer.weight.data)
                        if proj_layer.bias is not None:
                            torch.nn.init.zeros_(proj_layer.bias.data)
                        print(f"🎯 混合融合投影层已初始化: {attr_name}")
            
            print(f"🎯 改进的semantic embedding参数已初始化：{len(self.semantic_embeds)}个特征")
            if global_config.get('enable_codebook_reuse', True):
                print(f"    - 复用RQ码本权重，保持几何对齐，但是不一定是端到端")
            if hasattr(self, 'semantic_layer_weights') and len(self.semantic_layer_weights) > 0:
                print(f"    - 启用可学习per-layer权重：{len(self.semantic_layer_weights)}个特征")

        # 7. 时间调制模块初始化
        if hasattr(self, 'item_time_modulation') and self.item_time_modulation is not None:
            for module in self.item_time_modulation:
                if isinstance(module, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    torch.nn.init.zeros_(module.bias)
            # 门控参数已在创建时设为小值，保持渐进学习特性
            print("⏰ 时间调制模块权重已初始化")

        
        # 10. 显式初始化统一投影头（确保一致的初始化策略）
        for head_name in ['unified_item_seq_projection', 'unified_user_projection', 'unified_item_cand_projection']:
            head = getattr(self, head_name, None)
            if head is None:
                continue
            if isinstance(head, torch.nn.Linear):
                torch.nn.init.xavier_normal_(head.weight.data)
                if head.bias is not None:
                    torch.nn.init.zeros_(head.bias.data)
            elif isinstance(head, torch.nn.Sequential):
                for m in head:
                    if isinstance(m, torch.nn.Linear):
                        torch.nn.init.xavier_normal_(m.weight.data)
                        if m.bias is not None:
                            torch.nn.init.zeros_(m.bias.data)
        
        print("✅ BaselineModel权重初始化完成")

    def _get_adaptive_embedding_dim(self, feat_id, vocab_size):
        """
        根据特征ID和词汇表大小自适应选择embedding维度
        针对的是非连续特征（浮点数不能经过embedding）
        """
        import math
        
        # 获取自适应embedding配置
        embedding_config = get_embedding_config(self.args)
        
        # 如果未启用自适应，返回默认维度
        if not embedding_config.get('enable_adaptive_embedding', False):
            return embedding_config.get('sparse_embedding_dim', 64)
        
        # 检查特定特征覆盖配置
        per_feature_overrides = embedding_config.get('per_feature_overrides', {})
        if str(feat_id) in per_feature_overrides:
            override_config = per_feature_overrides[str(feat_id)]
            # 如果指定了out_dim，直接返回
            if 'out_dim' in override_config:
                if embedding_config.get('log_embedding_dim_decision', False):
                    print(f" 特征 {feat_id}: vocab={vocab_size}, dim={override_config['out_dim']} (per_feature_override)")
                return override_config['out_dim']
        
        # 时间稀疏特征专门处理
        time_config = embedding_config.get('time_features_config', {})
        if str(feat_id).startswith('time_') or feat_id == 'time_gap':
            dim = time_config.get('time_gap_embedding_dim', 16)
            if embedding_config.get('log_embedding_dim_decision', False):
                print(f"🚦 时间稀疏特征 {feat_id}: vocab={vocab_size}, dim={dim} (time_features_config)")
            return dim
        
        # 🎯 action_type 特征专门处理
        if feat_id == 'action_type':
            dim = time_config.get('action_type_embedding_dim', 8)  # 默认8维，比time_gap小一些
            if embedding_config.get('log_embedding_dim_decision', False):
                print(f"🎯 动作类型特征 {feat_id}: vocab={vocab_size}, dim={dim} (time_features_config)")
            return dim
        
        # 应用自适应维度公式
        formula_config = embedding_config.get('adaptive_dim_formula', {})
        _ratio = embedding_config.get('ratio', 1)
        k = formula_config.get('k', 8)
        alpha = formula_config.get('alpha', 0.25)
        min_dim = formula_config.get('min_dim', 8)
        max_dim = formula_config.get('max_dim', 96)
        
        # 特殊处理极小词表
        if vocab_size <= 32:
            dim = min(12, max(8, vocab_size // 3)) * _ratio
        elif vocab_size <= 128:
            dim = 16 * _ratio
        else:
            dim = int(k * (vocab_size ** alpha))
            dim = max(min_dim, min(dim, max_dim)) * _ratio
        
        if embedding_config.get('log_embedding_dim_decision', False):
            print(f"📏 特征 {feat_id}: vocab={vocab_size}, dim={dim} (adaptive_formula)")
        
        return dim

    def _init_feat_info(self, feat_statistics, feat_types):
        """
        将特征统计信息（特征数量）按特征类型分组产生不同的字典，方便声明稀疏特征的Embedding Table
        支持基于配置过滤被移除的特征

        Args:
            feat_statistics: 特征统计信息，key为特征ID，value为特征数量
            feat_types: 各个特征的特征类型，key为特征类型名称，value为包含的特征ID列表，包括user和item的sparse, array, emb, continual类型
        """
        # 获取要移除的特征列表
        embedding_config = get_embedding_config(self.args)
        per_feature_overrides = embedding_config.get('per_feature_overrides', {})
        removed_features = {k for k, v in per_feature_overrides.items() if v.get('action') == 'remove'}
        
        # 过滤被移除的特征
        self.USER_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types['user_sparse'] if k not in removed_features}
        self.USER_CONTINUAL_FEAT = feat_types['user_continual']
        self.ITEM_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types['item_sparse'] if k not in removed_features} 
        self.ITEM_CONTINUAL_FEAT = feat_types['item_continual']
        self.USER_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types['user_array'] if k not in removed_features}
        self.ITEM_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types['item_array'] if k not in removed_features}
        EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
        self.ITEM_EMB_FEAT = {k: EMB_SHAPE_DICT[k] for k in feat_types['item_emb']}  # 记录的是不同多模态特征的维度
        
        # 单独处理时间特征，避免影响候选item维度计算
        self.SEQ_TIME_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types.get('seq_time_sparse', []) if k not in removed_features}
        self.SEQ_TIME_CONTINUAL_FEAT = feat_types.get('seq_time_continual', [])
        
        # 🎯 action_type 现在通过 dataset.py 的 _init_feat_info 正确注册
        if 'action_type' in self.SEQ_TIME_SPARSE_FEAT:
            print(f"🎯 action_type 已从数据集注册，vocab_size={self.SEQ_TIME_SPARSE_FEAT['action_type']}")
        
        # 🎯 RQ-VAE预计算模式：添加semantic_id特征处理
        # semantic_81, semantic_82 ..
        self.ITEM_SEMANTIC_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types.get('item_semantic_array', []) if k not in removed_features}
        
        # 打印被移除的特征（如果有）
        if removed_features and embedding_config.get('log_embedding_dim_decision', False):
            print(f"🚫 已移除的特征: {removed_features}")

    def _monitor_batch_data(self, batch_data, mode='training'):
        """
        🔍 数据监控方法：检查batch数据的完整性和合理性
        
        Args:
            batch_data: 批次数据字典
            mode: 'training' 或 'inference'
        """
        if not self.enable_data_monitoring:
            return
            
        if mode == 'training':
            self.training_step_counter += 1
            should_monitor = (self.training_step_counter % self.monitoring_interval == 0)
        else:  # inference
            self.inference_batch_counter += 1
            should_monitor = (self.inference_batch_counter <= 3)  # 推理时只监控前3个batch
            
        if not should_monitor:
            return
            
        print(f"\n{'='*60}")
        print(f"🔍 数据监控 [{mode.upper()}] - Step {self.training_step_counter if mode == 'training' else self.inference_batch_counter}")
        print(f"{'='*60}")
        
        seq = batch_data.get('seq')
        feature_tensors = batch_data.get('feature_tensors', {})
        # 数据采样检查 - 显示具体数值
        if seq is not None:
            batch_size = seq.shape[0]
            sample_indices = torch.randperm(batch_size)[:min(self.monitoring_samples, batch_size)]
            
            print(f"🔬 样本数据检查 (显示{len(sample_indices)}个样本):")
            for i, idx in enumerate(sample_indices):
                print(f"  样本 {i+1} (batch_idx={idx.item()}):")
                
                # 序列ID
                seq_sample = seq[idx]
                non_zero_mask = seq_sample != 0
                non_zero_items = seq_sample[non_zero_mask]
                print(f"    seq长度: {non_zero_mask.sum().item()}/{seq_sample.shape[0]}")
                
                # 只取前5个item ID
                first_items = non_zero_items[:5]
                print(f"    seq前5项ID: {first_items.tolist()}")
                
                # 对应的embedding（只展示前5维）
                if hasattr(self, "item_embedding"):  
                    emb_vectors = self.item_embedding(first_items)  # shape=[k, dim]
                    for j, (item_id, emb_vec) in enumerate(zip(first_items.tolist(), emb_vectors)):
                        emb_preview = emb_vec[:5].detach().cpu().numpy().round(3).tolist()
                        print(f"      item {j+1} (ID={item_id}): embed前5维={emb_preview}")
                
                if mode == 'training':
                    pos_sample = batch_data.get('pos', torch.zeros_like(seq))[idx]
                    neg_sample = batch_data.get('neg', torch.zeros_like(seq))[idx]
                    pos_non_zero = pos_sample[pos_sample != 0]
                    neg_non_zero = neg_sample[neg_sample != 0]
                    print(f"    pos前3项: {pos_non_zero[:3].tolist()}")
                    print(f"    neg前3项: {neg_non_zero[:3].tolist()}")
                
                # 检查特征值范围
                if feature_tensors:
                    print(f"    特征样本值:")
                    for feat_name, tensor in list(feature_tensors.items()): 
                        if len(tensor.shape) >= 2:
                            feat_sample = tensor[idx]
                            if feat_sample.numel() > 0:
                                min_val = feat_sample.min().item()
                                max_val = feat_sample.max().item()
                                non_zero_count = (feat_sample != 0).sum().item()
                                print(f"      {feat_name}: range=[{min_val:.3f}, {max_val:.3f}], non_zero={non_zero_count}/{feat_sample.numel()}")
                                
                                # 🕐 专门打印时间戳特征的详细信息
                                if 'timestamp' in feat_name.lower() or feat_name in ['seq_timestamp', 'pos_timestamp', 'neg_timestamp']:
                                    print(f"      📅 时间戳特征 {feat_name} 详细信息:")
                                    # 显示前10个时间戳值
                                    timestamp_values = feat_sample[:10].detach().cpu().numpy() if feat_sample.numel() >= 10 else feat_sample.detach().cpu().numpy()
                                    print(f"        前{len(timestamp_values)}个时间戳值: {timestamp_values.tolist()}")
                                    # 计算时间戳统计信息
                                    valid_timestamps = feat_sample[feat_sample > 0]
                                    if valid_timestamps.numel() > 0:
                                        mean_ts = valid_timestamps.float().mean().item()
                                        std_ts = valid_timestamps.float().std().item() if valid_timestamps.numel() > 1 else 0.0
                                        print(f"        有效时间戳统计: 数量={valid_timestamps.numel()}, 均值={mean_ts:.2f}, 标准差={std_ts:.2f}")
                                        # 时间戳间隔分析
                                        if valid_timestamps.numel() > 1:
                                            time_diffs = valid_timestamps[1:] - valid_timestamps[:-1]
                                            avg_interval = time_diffs.float().mean().item()
                                            print(f"        时间间隔分析: 平均间隔={-avg_interval:.2f}")
                                    else:
                                        print(f"        ⚠️ 该样本无有效时间戳")
        
        # 4. 异常值检测
        print(f"⚠️  异常检测:")
        anomalies = []
        
        if seq is not None:
            # 检查序列ID范围
            seq_max = seq.max().item()
            seq_min = seq.min().item()
            if seq_max > self.item_num or seq_min < 0:
                anomalies.append(f"seq ID范围异常: [{seq_min}, {seq_max}], 期望[0, {self.item_num}]")
        
        # 检查特征tensor中的NaN/Inf
        for feat_name, tensor in feature_tensors.items():
            if torch.isnan(tensor).any():
                anomalies.append(f"{feat_name}包含NaN")
            if torch.isinf(tensor).any():
                anomalies.append(f"{feat_name}包含Inf")
        
        if anomalies:
            for anomaly in anomalies:
                print(f"    ❌ {anomaly}")
        else:
            print(f"    ✅ 未发现明显异常")
        
        print(f"{'='*60}\n")

    def feat2emb(self, seq, feature_tensors, mask=None, include_user=False, mode='seq'):
        """
        ⚡ 特征embedding统一入口：支持Field-wise投影和传统模式
        
        Args:
            seq: 序列ID，shape为 [batch_size, maxlen]
            feature_tensors: 从Dataset传来的已处理特征tensor字典
            mask: 掩码，1表示item，2表示user
            include_user: 是否处理用户特征
            mode: 'seq'表示序列token(含时间), 'cand'表示候选item(无时间)

        Returns:
            seqs_emb: 序列特征的Embedding
        """
        return self._feat2emb_dispatcher(seq, feature_tensors, mask, include_user, mode)

    def _process_semantic_feature(self, feat_id, ids_tensor):
        """
        🎯 改进的语义ID处理：明确区分不同融合模式的处理逻辑
        
        主要改进：
        1. 明确区分各融合模式的适用场景和权重使用
        2. 清晰的模式隔离，避免逻辑混淆
        3. 复用特征仅在端到端模式+权重复用开启时生效
        4. learnable权重仅在weighted_sum模式下使用
        """
        # 🔄 定义通用的投影前向函数，供checkpoint使用
        def projection_forward(x, proj_layer):
            return proj_layer(x)
        ids = ids_tensor.to(self.dev).long()
        
        # 解析配置
        semantic_config = get_semantic_id_config()
        feature_id = feat_id.split('_')[1] if '_' in feat_id else feat_id  # '81'
        cfg = semantic_config['semantic_id_features'].get(feature_id, None)
        if cfg is None:
            return None
        
        feature_name = cfg['feature_name']  # semantic_81
        num_codebooks = cfg['array_length']
        padding_value = cfg.get('padding_value', 0)
        fusion_mode = self.semantic_fusion_mode.get(feature_name, 'sum')
        
        if not hasattr(self, 'semantic_embeds') or feature_name not in self.semantic_embeds:
            return None
        
        B, L, M = ids.shape
        assert M == num_codebooks, f"{feature_name} 的codebook数不匹配: ids={M}, cfg={num_codebooks}"
        
        # 1. 分别获取所有codebook的embedding
        embeddings = []
        for m in range(num_codebooks):
            emb_table = self.semantic_embeds[feature_name][m]
            ids_m = ids[..., m].clamp(min=0, max=emb_table.num_embeddings - 1)  # 确保ID有效
            emb_m = emb_table(ids_m)  # [B, L, D]
            embeddings.append(emb_m)
        
        # 2. 🎯 改进的融合策略：明确区分各模式
        if fusion_mode == 'sum':
            # 等权求和：与RQ-VAE的z_q = ∑e_m[idx_m]完全一致，无可学习权重
            fused_semantic = torch.stack(embeddings, dim=0).sum(dim=0)  # [B, L, D]
            
        elif fusion_mode == 'weighted_sum':
            # 加权求和：使用可学习的per-layer权重（仅此模式使用learnable权重）
            if feature_name in self.semantic_layer_weights:
                weights = F.softmax(self.semantic_layer_weights[feature_name], dim=0)  # [M]
                weighted_embeddings = [w * emb for w, emb in zip(weights, embeddings)]
                fused_semantic = torch.stack(weighted_embeddings, dim=0).sum(dim=0)  # [B, L, D]
            else:
                # 如果没有权重则fallback到等权求和，并给出警告
                print(f"⚠️ 警告：{feature_name}配置为weighted_sum但无可学习权重，fallback到sum模式")
                fused_semantic = torch.stack(embeddings, dim=0).sum(dim=0)
                
        elif fusion_mode == 'concat':
            # 拼接模式：传统向量拼接（向后兼容），无权重概念
            fused_semantic = torch.cat(embeddings, dim=-1)  # [B, L, M*D]
            
        elif fusion_mode == 'hybrid':
            # 混合模式：sum分支 + concat分支的可学习加权组合
            sum_branch = torch.stack(embeddings, dim=0).sum(dim=0)  # [B, L, D]
            concat_branch = torch.cat(embeddings, dim=-1)  # [B, L, M*D]
            
            # 投影concat分支到与sum分支相同的维度
            if not hasattr(self, f'_hybrid_proj_{feature_name}'):
                concat_dim = embeddings[0].shape[-1] * num_codebooks
                sum_dim = embeddings[0].shape[-1]
                proj_layer = torch.nn.Linear(concat_dim, sum_dim).to(self.dev)
                setattr(self, f'_hybrid_proj_{feature_name}', proj_layer)
                
            proj_layer = getattr(self, f'_hybrid_proj_{feature_name}')
            # Hybrid投影层支持checkpoint
            if self.training and self.enable_projection_checkpoint:
                concat_branch_proj = checkpoint(projection_forward, concat_branch, proj_layer, use_reentrant=False)
            else:
                concat_branch_proj = proj_layer(concat_branch)  # [B, L, D]
            
            # 使用可学习的混合权重
            hybrid_weight_key = f"{feature_name}_hybrid"
            if hybrid_weight_key in self.semantic_layer_weights:
                hybrid_weight = torch.sigmoid(self.semantic_layer_weights[hybrid_weight_key])
            else:
                # fallback到配置的固定权重
                global_config = semantic_config.get('rqvae_alignment', {})
                hybrid_weight = global_config.get('hybrid_fusion_weight', 0.7)
                
            fused_semantic = hybrid_weight * sum_branch + (1 - hybrid_weight) * concat_branch_proj
            
        else:
            raise ValueError(f"不支持的融合模式: {fusion_mode}")
        
        # 3. 🎯 融合后归一化（可选）：稳定表示范数/分布
        if self.enable_post_fusion_norm:
            if self.post_fusion_norm_type == 'layernorm':
                if feature_name in self.semantic_norms:
                    fused_semantic = self.semantic_norms[feature_name](fused_semantic)
            elif self.post_fusion_norm_type == 'l2':
                fused_semantic = F.normalize(fused_semantic, p=2, dim=-1, eps=self.post_fusion_norm_eps)
        
        # 4. 🎯 可选的SID dropout（仅对求和类模式有效，拼接模式不适用）
        if (self.training and self.enable_sid_dropout and 
            self.sid_dropout_rate > 0 and fusion_mode in ['sum', 'weighted_sum', 'hybrid']):
            if torch.rand(1).item() < self.sid_dropout_rate:
                # 以一定概率将整个语义向量置零，迫使模型依赖其他特征
                fused_semantic = torch.zeros_like(fused_semantic)

        return fused_semantic

    def _feat2emb_dispatcher(self, seq, feature_tensors, mask=None, include_user=False, mode='seq'):
        """
        ⚡ 特征embedding分发器：根据配置选择Field-wise投影或传统模式
        同时处理端到端RQ-VAE的语义ID转换
        
        Args:
            seq: 序列ID，shape为 [batch_size, maxlen]
            feature_tensors: 从Dataset传来的已处理特征tensor字典
            mask: 掩码，1表示item，2表示user
            include_user: 是否处理用户特征
            mode: 'seq'表示序列token(含时间), 'cand'表示候选item(无时间)

        Returns:
            seqs_emb: 序列特征的Embedding
        """
        seq = seq.to(self.dev)
        
        # 🎯 端到端模式：实时将原始多模态特征转换为统一的语义ID数组格式
        if (self.enable_rqvae and not self.use_precomputed_semantic_ids and 
            self.rqvae_models and self.original_emb_feat):
            for prefix_name in ['seq_', 'pos_', 'neg_']:
                for feat_id in self.original_emb_feat.keys():
                    tensor_key = f'{prefix_name}{feat_id}'
                    if tensor_key in feature_tensors:
                        raw_feature = feature_tensors[tensor_key].to(self.dev)
                        batch_size, maxlen, emb_dim = raw_feature.shape
                        flat_feature = raw_feature.reshape(-1, emb_dim)
                        if feat_id in self.rqvae_models:
                            rqvae_model = self.rqvae_models[feat_id]
                            with torch.no_grad():
                                semantic_ids = rqvae_model._get_codebook(flat_feature)  # [N, num_codebooks]
                                semantic_feat_name = f"semantic_{feat_id}"
                                if hasattr(self, 'semantic_embeds') and semantic_feat_name in self.semantic_embeds:
                                    vocab_size = self.semantic_embeds[semantic_feat_name][0].num_embeddings
                                    semantic_ids = torch.clamp(semantic_ids, 0, vocab_size - 1)
                                # 🎯 转换为统一的数组格式
                                semantic_array = semantic_ids.reshape(batch_size, maxlen, -1)  # [batch_size, maxlen, num_codebooks]
                                # 注册为统一的语义ID特征
                                feature_tensors[f'{prefix_name}{semantic_feat_name}'] = semantic_array

        return self._feat2emb_traditional(seq, feature_tensors, mask, include_user, mode)

    def _feat2emb_traditional(self, seq, feature_tensors, mask=None, include_user=False, mode='seq'):
        """
        传统模式的特征embedding处理（保持原有逻辑）
        """
        # 🔄 定义通用的投影前向函数，供所有checkpoint使用
        def projection_forward(x, proj_layer):
            return proj_layer(x)
        seq = seq.to(self.dev)
        
        # 📏 ID embedding
        if include_user:
            user_mask = (mask == 2).to(self.dev)
            item_mask = (mask == 1).to(self.dev)
            user_embedding = self.user_emb(user_mask * seq)
            item_embedding = self.item_emb(item_mask * seq)
            
            # 🔥 新增：ID Dropout - 仅在训练时中对item embedding进行dropout，user 没有冷启动
            if self.training and self.drop_id_rate > 0:
                item_embedding = self._apply_id_dropout(item_embedding)
            
            item_feat_list = [item_embedding]
            user_feat_list = [user_embedding]
        else:
            item_embedding = self.item_emb(seq)  
            # 🔥 新增：ID Dropout - 仅在训练时中对item embedding进行dropout，user 没有冷启动
            if self.training and self.drop_id_rate > 0:
                item_embedding = self._apply_id_dropout(item_embedding)
            
            item_feat_list = [item_embedding]

        # 始终使用seq_前缀
        prefix = 'seq_'  
        
        # 📏 处理item稀疏特征 - 使用原始embedding维度
        for feat_id in self.ITEM_SPARSE_FEAT:
            if feat_id not in self.sparse_emb:
                continue
            tensor_key = f'{prefix}{feat_id}'
            if tensor_key in feature_tensors:
                tensor_feature = feature_tensors[tensor_key].to(self.dev)
                sparse_emb = self.sparse_emb[feat_id](tensor_feature)
                item_feat_list.append(sparse_emb)
        
        # 📏 处理item数组特征 - 使用原始embedding维度
        for feat_id in self.ITEM_ARRAY_FEAT:
            if feat_id not in self.sparse_emb:
                continue
            tensor_key = f'{prefix}{feat_id}'
            if tensor_key in feature_tensors:
                tensor_feature = feature_tensors[tensor_key].to(self.dev)
                array_emb = self.sparse_emb[feat_id](tensor_feature).sum(2)
                item_feat_list.append(array_emb)
        
        # 处理item连续特征 - 应用投影层
        if len(self.ITEM_CONTINUAL_FEAT) > 0:
            continual_feats = []
            for feat_id in self.ITEM_CONTINUAL_FEAT:
                tensor_key = f'{prefix}{feat_id}'
                if tensor_key in feature_tensors:
                    tensor_feature = feature_tensors[tensor_key].to(self.dev)
                    if tensor_feature.dim() == 2:
                        tensor_feature = tensor_feature.unsqueeze(-1)
                    continual_feats.append(tensor_feature)
            
            if continual_feats and hasattr(self, 'item_continual_projection') and self.item_continual_projection is not None:
                all_continual_feats = torch.cat(continual_feats, dim=-1)
                assert all_continual_feats.shape[-1] == self.item_continual_projection.in_features, \
                    f"Item连续特征维度不匹配: 期望{self.item_continual_projection.in_features}, 实际{all_continual_feats.shape[-1]}"
                # Item连续特征投影支持checkpoint
                if self.training and self.enable_continual_projection_checkpoint:
                    projected_continual = checkpoint(projection_forward, all_continual_feats, self.item_continual_projection, use_reentrant=False)
                else:
                    projected_continual = self.item_continual_projection(all_continual_feats)
                item_feat_list.append(projected_continual)
        
        # 📏 处理用户特征（如果需要）
        if include_user:
            for feat_id in self.USER_SPARSE_FEAT:
                if feat_id not in self.sparse_emb:
                    continue
                tensor_key = f'{prefix}{feat_id}'
                if tensor_key in feature_tensors:
                    tensor_feature = feature_tensors[tensor_key].to(self.dev)
                    sparse_emb = self.sparse_emb[feat_id](tensor_feature)
                    user_feat_list.append(sparse_emb)
            
            for feat_id in self.USER_ARRAY_FEAT:
                if feat_id not in self.sparse_emb:
                    continue
                tensor_key = f'{prefix}{feat_id}'
                if tensor_key in feature_tensors:
                    tensor_feature = feature_tensors[tensor_key].to(self.dev)
                    array_emb = self.sparse_emb[feat_id](tensor_feature).sum(2)
                    user_feat_list.append(array_emb)
            
            # 处理用户连续特征 - 应用投影层
            if len(self.USER_CONTINUAL_FEAT) > 0:
                continual_feats = []
                for feat_id in self.USER_CONTINUAL_FEAT:
                    tensor_key = f'{prefix}{feat_id}'
                    if tensor_key in feature_tensors:
                        tensor_feature = feature_tensors[tensor_key].to(self.dev)
                        if tensor_feature.dim() == 2:
                            tensor_feature = tensor_feature.unsqueeze(-1)
                        continual_feats.append(tensor_feature)
                
                if continual_feats and hasattr(self, 'user_continual_projection') and self.user_continual_projection is not None:
                    all_continual_feats = torch.cat(continual_feats, dim=-1)
                    # 🎯 强制要求维度匹配，确保连续特征经过正确的线性投影
                    assert all_continual_feats.shape[-1] == self.user_continual_projection.in_features, \
                        f"User连续特征维度不匹配: 期望{self.user_continual_projection.in_features}, 实际{all_continual_feats.shape[-1]}"
                    # User连续特征投影支持checkpoint
                    if self.training and self.enable_continual_projection_checkpoint:
                        projected_continual = checkpoint(projection_forward, all_continual_feats, self.user_continual_projection, use_reentrant=False)
                    else:
                        projected_continual = self.user_continual_projection(all_continual_feats)
                    user_feat_list.append(projected_continual)

        # 🎯 统一多模态特征处理：传统模式 vs RQ-VAE模式
        if not self.enable_rqvae:
            for feat_id in self.ITEM_EMB_FEAT:
                tensor_key = f'{prefix}{feat_id}'
                if tensor_key in feature_tensors:
                    tensor_feature = feature_tensors[tensor_key].to(self.dev)
                    transformed_feature = self.emb_transform[feat_id](tensor_feature)
                    item_feat_list.append(transformed_feature)
        else:
            # 🎯 RQ-VAE预处理（预计算语义ID）：按 codebook 独立Embedding + 加权求和 + LayerNorm
            for feat_id in self.ITEM_SEMANTIC_ARRAY_FEAT:
                tensor_key = f'{prefix}{feat_id}'
                if tensor_key not in feature_tensors:
                    continue

                fused_semantic = self._process_semantic_feature(feat_id, feature_tensors[tensor_key])
                if fused_semantic is not None:
                    item_feat_list.append(fused_semantic)

        # 🕐 条件性添加时间特征：仅对序列token(mode='seq')添加
        if mode == 'seq' and self.enable_time_features:
            # 相对时间特征(time_gap)通过SEQ_TIME_SPARSE_FEAT处理
            for feat_id in self.SEQ_TIME_SPARSE_FEAT:
                tensor_key = f'seq_{feat_id}'
                if tensor_key in feature_tensors:
                    time_tensor = feature_tensors[tensor_key].to(self.dev)
                    time_emb = self.sparse_emb[feat_id](time_tensor)
                    item_feat_list.append(time_emb)
            
            # 🎯 改进：动态处理所有连续时间特征
            if len(self.SEQ_TIME_CONTINUAL_FEAT) > 0:
                continual_time_features = []
                for feat_id in self.SEQ_TIME_CONTINUAL_FEAT:
                    tensor_key = f'seq_{feat_id}'
                    if tensor_key in feature_tensors:
                        feat_tensor = feature_tensors[tensor_key].to(self.dev)
                        if feat_tensor.dim() == 2:
                            feat_tensor = feat_tensor.unsqueeze(-1)
                        continual_time_features.append(feat_tensor)
                
                if continual_time_features:
                    all_continual_time_concat = torch.cat(continual_time_features, dim=-1)
                    if hasattr(self, 'absolute_time_projection') and self.absolute_time_projection is not None:
                        # 🎯 强制要求维度匹配，确保时间连续特征经过正确的线性投影
                        assert all_continual_time_concat.shape[-1] == self.absolute_time_projection.in_features, \
                            f"时间连续特征维度不匹配: 期望{self.absolute_time_projection.in_features}, 实际{all_continual_time_concat.shape[-1]}"
                        # 时间特征投影支持checkpoint
                        if self.training and self.enable_continual_projection_checkpoint:
                            absolute_time_emb = checkpoint(projection_forward, all_continual_time_concat, self.absolute_time_projection, use_reentrant=False)
                        else:
                            absolute_time_emb = self.absolute_time_projection(all_continual_time_concat)
                        item_feat_list.append(absolute_time_emb)
                    # 如果没有投影层，说明没有时间连续特征，跳过处理

        # 🎯 简化特征融合：直接拼接+统一投影
        if len(item_feat_list) > 0:
            # 检查并统一seq_len维度
            target_seq_len = item_feat_list[0].shape[1]
            for i, feat in enumerate(item_feat_list):
                if feat.shape[1] != target_seq_len:
                    print(f"⚠️ 警告：特征{i}的seq_len维度不匹配，期望{target_seq_len}，实际{feat.shape[1]}")
                    if feat.shape[1] == 1:
                        item_feat_list[i] = feat.expand(-1, target_seq_len, -1)
                    elif target_seq_len == 1:
                        item_feat_list[i] = feat[:, :1, :]
            
            # 简单拼接所有特征
            all_item_emb = torch.cat(item_feat_list, dim=2)
            in_feat_expected = self.unified_item_seq_projection.in_features if mode=='seq' else self.unified_item_cand_projection.in_features
            if all_item_emb.shape[-1] != in_feat_expected:
                print("❌ 统一投影维度不匹配调试信息：")
                print(f" mode={mode}")
                print(f"  当前拼接维度={all_item_emb.shape[-1]}, 期望={in_feat_expected}")
                detail_dims = [t.shape[-1] for t in item_feat_list]
                print(f"  子特征维度列表={detail_dims}")
                # 尝试列出注册的时间稀疏特征embedding维度
                try:
                    embedding_config = get_embedding_config(self.args)
                    time_config = embedding_config.get('time_features_config', {})
                    print(f"  配置: time_gap_dim={time_config.get('time_gap_embedding_dim', 'NA')}, action_type_dim={time_config.get('action_type_embedding_dim', 8)}")
                    print(f"  注册稀疏时间特征={list(self.SEQ_TIME_SPARSE_FEAT.keys())}")
                except Exception as _e:
                    print(f"  (无法加载embedding_config: {_e})")
                raise RuntimeError(f"统一投影in_features不匹配: got {all_item_emb.shape[-1]} expect {in_feat_expected}")
            # 根据mode选择相应的统一投影层，支持checkpoint
            if mode == 'seq':
                if self.training and self.enable_projection_checkpoint:
                    all_item_emb = checkpoint(projection_forward, all_item_emb, self.unified_item_seq_projection, use_reentrant=False)
                else:
                    all_item_emb = self.unified_item_seq_projection(all_item_emb)
            else:
                if self.training and self.enable_projection_checkpoint:
                    all_item_emb = checkpoint(projection_forward, all_item_emb, self.unified_item_cand_projection, use_reentrant=False)
                else:
                    all_item_emb = self.unified_item_cand_projection(all_item_emb)
        else:
            # 如果没有特征，创建零向量
            batch_size = seq.shape[0]
            seq_len = seq.shape[1]
            all_item_emb = torch.zeros(batch_size, seq_len, self.hidden_units, device=self.dev)
        
        # 🎯 标准化特征后再应用Dropout：LayerNorm → Dropout顺序
        all_item_emb = self.emb_layernorm(all_item_emb)  # 先LayerNorm，在完整分布上归一化
        # if mode == 'seq':  # 只对序列编码使用embedding dropout，候选项不使用
        #     all_item_emb = self.emb_dropout(all_item_emb)  # 后Dropout，避免破坏归一化效果
        
        if include_user:
            if len(user_feat_list) > 0:
                all_user_emb = torch.cat(user_feat_list, dim=2)
                # 用户投影层也支持checkpoint
                if self.training and self.enable_projection_checkpoint:
                    all_user_emb = checkpoint(projection_forward, all_user_emb, self.unified_user_projection, use_reentrant=False)
                else:
                    all_user_emb = self.unified_user_projection(all_user_emb)
            else:
                batch_size = seq.shape[0]
                seq_len = seq.shape[1]
                all_user_emb = torch.zeros(batch_size, seq_len, self.hidden_units, device=self.dev)
            all_user_emb = self.emb_layernorm(all_user_emb)  # 先LayerNorm，在完整分布上归一化
            # if mode == 'seq':  # 只对序列编码使用embedding dropout
            #     all_user_emb = self.emb_dropout(all_user_emb)  # 后Dropout，避免破坏归一化效果
            seqs_emb = all_item_emb + all_user_emb
        else:
            seqs_emb = all_item_emb
            
        return seqs_emb

    def forward_sequence(self, log_seqs, mask, feature_tensors):
        """
        ⚡ 多进程优化版本：与训练路径一致的序列前向，返回整段序列隐表示 [batch_size, seq_len, hidden_units]
        说明：用于验证阶段的ranking指标计算，等价于log2feats，提供清晰语义接口
        """
        return self.log2feats(log_seqs, mask, feature_tensors)

    def log2feats(self, log_seqs, mask, feature_tensors):
        """
        ⚡ 多进程优化版本：直接接收tensor字典
        Args:
            log_seqs: 序列ID
            mask: token类型掩码，1表示item token，2表示user token
            feature_tensors: 从Dataset传来的已处理特征tensor字典
        Returns:
            seqs_emb: 序列的Embedding，形状为 [batch_size, maxlen, hidden_units]
        """
        # log2feats 开头
        log_seqs = log_seqs.to(self.dev)
        mask      = mask.to(self.dev)           # 后面也会参与 attention_mask 计算
        batch_size = log_seqs.shape[0]
        maxlen = log_seqs.shape[1]
        seqs = self.feat2emb(log_seqs, feature_tensors, mask=mask, include_user=True, mode='seq') # 将序列中的特征转换为Embedding，shape为 [batch_size, maxlen, hidden_units]
        # seqs *= self.hidden_units**0.5  # LayerNorm后不需要额外缩放，避免破坏归一化效果
        seqs = self.emb_dropout(seqs)  # Embedding Dropout
        
        # 🕐 提取时间戳用于成对时间偏置和时间衰减权重
        seq_timestamps = None
        ts_valid_mask = None
        if (self.enable_time_features and 'seq_timestamp' in feature_tensors):
            try:
                seq_timestamps = feature_tensors['seq_timestamp'].to(self.dev)  # [batch_size, maxlen]
                # 确保时间戳的形状与序列一致
                if seq_timestamps.shape[0] != batch_size or seq_timestamps.shape[1] != maxlen:
                    print(f"⚠️ 时间戳形状不匹配: seq_timestamps={seq_timestamps.shape}, expected=[{batch_size}, {maxlen}]")
                    seq_timestamps = None
            except Exception as e:
                print(f"⚠️ 时间戳提取失败: {e}")
                seq_timestamps = None
        
        # 时间戳有效性掩码（>0 视为有效）
        if seq_timestamps is not None:
            ts_valid_mask = (seq_timestamps > 0)
        
        # 🎯 HSTU改进：智能位置编码应用
        if self.use_absolute_pos_emb and self.pos_emb is not None:
            # 仅在没有RoPE和相对位置偏置时使用传统绝对位置编码
            poss = torch.arange(1, maxlen + 1, device=self.dev).unsqueeze(0).expand(batch_size, -1).clone()
            poss *= log_seqs != 0
            seqs += self.pos_emb(poss)

        maxlen = seqs.shape[1]
        ones_matrix = torch.ones((maxlen, maxlen), dtype=torch.bool, device=self.dev)
        attention_mask_tril = torch.tril(ones_matrix)
        attention_mask_pad = (mask != 0).to(self.dev)
        attention_mask = attention_mask_tril.unsqueeze(0) & attention_mask_pad.unsqueeze(1)

        # 🎯 修复：使用seq_action_type作为时间偏置门控的token_type，而非item/user的mask
        seq_action_type_tensor = None
        if (self.enable_time_bias and 'seq_action_type' in feature_tensors):
            try:
                seq_action_type_tensor = feature_tensors['seq_action_type'].to(self.dev)  # [batch_size, maxlen]
                # 确保动作类型的形状与序列一致
                if seq_action_type_tensor.shape[0] != batch_size or seq_action_type_tensor.shape[1] != maxlen:
                    print(f"⚠️ 动作类型形状不匹配: seq_action_type={seq_action_type_tensor.shape}, expected=[{batch_size}, {maxlen}]")
                    seq_action_type_tensor = None
            except Exception as e:
                print(f"⚠️ 动作类型提取失败: {e}")
                seq_action_type_tensor = None
        
        for i in range(len(self.attention_layers)):
            if self.norm_first: # Pre-Norm: 推荐默认模式
                x = self.attention_layernorms[i](seqs) 
                mha_outputs, _ = self.attention_layers[i](x, x, x, attn_mask=attention_mask, seq_timestamps=seq_timestamps, token_type=seq_action_type_tensor, ts_valid_mask=ts_valid_mask)
                seqs = seqs + mha_outputs 
                ffn_input = self.forward_layernorms[i](seqs)
                ffn_output = self.forward_layers[i](ffn_input)
                seqs = seqs + ffn_output
            else: # Post-Norm: 修复双重归一化问题
                mha_outputs, _ = self.attention_layers[i](seqs, seqs, seqs, attn_mask=attention_mask, seq_timestamps=seq_timestamps, token_type=seq_action_type_tensor, ts_valid_mask=ts_valid_mask)
                seqs = self.attention_layernorms[i](seqs + mha_outputs)
                ffn_outputs = self.forward_layers[i](seqs)
                seqs = self.forward_layernorms[i](seqs + ffn_outputs)
        # Pre-Norm需要最后一层LN，Post-Norm已经在块内归一化完毕
        log_feats = self.last_layernorm(seqs) if self.norm_first else seqs
        return log_feats

    def forward(self, batch_data):
        """
        ⚡ 多进程优化版本：训练时调用，返回序列、正样本、负样本的embeddings，用于InfoNCE loss计算

        Args:
            batch_data: 从Dataset.collate_fn传来的完整batch字典

        Returns:
            seq_embs: 序列嵌入，形状为 [batch_size, maxlen, hidden_units]
            pos_embs: 正样本嵌入，形状为 [batch_size, maxlen, hidden_units]
            neg_embs: 负样本嵌入，形状为 [batch_size, maxlen, hidden_units]
            loss_mask: 损失掩码，形状为 [batch_size, maxlen]
            pos_seqs: 正样本ID
            neg_seqs: 负样本ID  
            next_action_type: 下一个token动作类型；下一个token动作类型，0表示曝光，1表示点击
        """
        # 🔍 数据监控 - 训练模式
        self._monitor_batch_data(batch_data, mode='training')
        
        # 解包batch数据
        user_item = batch_data['seq']
        pos_seqs = batch_data['pos']
        neg_seqs = batch_data['neg']
        mask = batch_data['token_type']
        next_mask = batch_data['next_token_type']
        next_action_type = batch_data['next_action_type']
        feature_tensors = batch_data['feature_tensors']
        
        seq_embs = self.log2feats(user_item, mask, feature_tensors)
        next_action_type = next_action_type.to(self.dev)
        next_mask = next_mask.to(self.dev)
        loss_mask = (next_mask == 1)

        # 为pos和neg特征创建正确的tensor字典
        pos_feature_tensors = {k.replace('pos_', 'seq_'): v for k, v in feature_tensors.items() if k.startswith('pos_')}
        neg_feature_tensors = {k.replace('neg_', 'seq_'): v for k, v in feature_tensors.items() if k.startswith('neg_')}

        pos_embs = self.feat2emb(pos_seqs, pos_feature_tensors, include_user=False, mode='cand')
        neg_embs = self.feat2emb(neg_seqs, neg_feature_tensors, include_user=False, mode='cand')

        return seq_embs, pos_embs, neg_embs, loss_mask, pos_seqs, neg_seqs, next_action_type

    def predict(self, batch_data):
        """
        计算用户序列的表征 - 使用L2归一化确保与训练时的余弦相似度一致
        Args:
            batch_data: 从Dataset.collate_fn传来的测试batch字典
        Returns:
            final_feat: 用户序列的L2归一化表征，形状为 [batch_size, hidden_units]
        """
        # 🔍 数据监控 - 推理模式
        self._monitor_batch_data(batch_data, mode='inference')
        
        log_seqs = batch_data['seq']
        mask = batch_data['token_type']
        feature_tensors = batch_data['feature_tensors'] # 特征字典
        
        log_feats = self.log2feats(log_seqs, mask, feature_tensors)
        final_feat = log_feats[:, -1, :]
        
        # L2 normalize for consistency with InfoNCE training (cosine similarity)
        final_feat = final_feat / final_feat.norm(dim=-1, keepdim=True)

        return final_feat

    def compute_infonce_loss(self, seq_embs, pos_embs, neg_embs, loss_mask, pos_seqs=None, neg_seqs=None, next_action_type=None, exposure_weight=0.6, writer=None, global_step=None, chunk_size=2048, enable_chunked_computation=False, return_detailed_loss=False):
        """
        改进的InfoNCE损失计算 - 支持 in-batch 负样本 + 全局负样本混合
        主要改进：
        1. 🎯 In-batch负样本：使用其他样本的正样本作为当前样本的负样本（热门样本为主）
        3. 🎯 行级自正样本屏蔽：避免将自己的正样本作为自己的负样本
        4. 真正的流式分块计算：不物化完整neg_logits矩阵，直接分块求loss并累计
        5. False Negative过滤：从负样本池中移除batch内其他用户的正样本
        6. 样本级加权：点击(权重1.0) + 曝光(可配置权重，默认0.6)，按样本数加权而非类别数平均
        7. 数值稳定性：统一温度缩放，处理边界情况
        Args:
            seq_embs: 序列嵌入，形状为 [batch_size, maxlen, hidden_units]
            pos_embs: 正样本嵌入，形状为 [batch_size, maxlen, hidden_units]  
            neg_embs: 负样本嵌入，形状为 [batch_size, maxlen, hidden_units]
            loss_mask: 损失掩码，形状为 [batch_size, maxlen]
            pos_seqs: 正样本ID序列，形状为 [batch_size, maxlen]，用于构建全局正样本集合
            neg_seqs: 负样本ID序列，形状为 [batch_size, maxlen]，用于过滤false negative
            next_action_type: 动作类型，形状为 [batch_size, maxlen]，0表示曝光，1表示点击
            exposure_weight: 曝光样本权重(0.0-1.0)，默认0.6，平衡曝光和点击的重要性
            writer: TensorBoard writer for logging (optional)
            global_step: Global step for logging (optional)
            chunk_size: 分块大小，默认2048
            enable_chunked_computation: 是否启用分块计算
            return_detailed_loss: 是否返回详细的损失分解 (total_loss, click_loss, expo_loss)
        Returns:
            如果 return_detailed_loss=True: (total_loss, click_loss, expo_loss)
            否则: total_loss
        """
            
        batch_size, maxlen, hidden_size = seq_embs.shape
        
        # 用官方实现进行 L2 归一化（带 eps 防止除零）
        eps = 1e-5
        seq_embs = F.normalize(seq_embs, p=2, dim=-1, eps=eps)
        pos_embs = F.normalize(pos_embs, p=2, dim=-1, eps=eps)
        neg_embs = F.normalize(neg_embs, p=2, dim=-1, eps=eps)
 
        # 2. 负样本池处理 
        neg_embedding_all = neg_embs.reshape(-1, hidden_size) # [batch_size*maxlen, hidden_size]
        
        # 🎯 直接使用原始负样本池，不做复杂过滤
        clean_neg_embedding = neg_embedding_all
        
        # === 🎯 直接拼接所有有效正样本，计算时掩码避免自对比 ===
        inbatch_info = None  # 存储 in-batch 信息，供后续掩码使用
        
        if self.enable_inbatch_negatives:
            loss_mask_flat = loss_mask.reshape(-1)  # [batch_size*maxlen]
            valid_indices = torch.nonzero(loss_mask_flat, as_tuple=False).squeeze(-1) #[num_valid]
            num_valid = valid_indices.numel() # []
            
            if num_valid >= 2:  # 至少2个有效位置
                pos_flat = pos_embs.reshape(-1, hidden_size)  # [batch_size*maxlen, hidden_size]
                valid_pos_embs = pos_flat[valid_indices]  # [num_valid, hidden_size]
                
                # 直接拼接所有有效正样本作为额外负样本（最高效）
                final_neg_embedding = torch.cat([clean_neg_embedding, valid_pos_embs], dim=0)
                
                # 一次性构建 index_map：index_map[flat_pos] = 其在 valid_indices 中的列号，其余为 -1
                index_map = torch.full(
                    (loss_mask_flat.size(0),), -1,
                    device=loss_mask_flat.device, dtype=torch.long
                )
                index_map[valid_indices] = torch.arange(num_valid, device=loss_mask_flat.device)
                
                # 保存信息供后续掩码使用
                inbatch_info = {
                    'valid_indices': valid_indices,  # 有效位置在扁平化序列中的索引
                    'num_global_neg': clean_neg_embedding.shape[0],  # 全局负样本数量
                    'num_valid': num_valid,  # 有效位置数量
                    'index_map': index_map  # 位置到列号的映射
                }
            else:
                final_neg_embedding = clean_neg_embedding
        else:
            final_neg_embedding = clean_neg_embedding
        
        # 🔧 真正的分块计算：流式求loss，不物化完整neg_logits矩阵
        if enable_chunked_computation:
            # 设置默认分块大小
            if chunk_size is None:
                chunk_size = getattr(self.args, 'infonce_chunk_size', 2048)
            
            # 准备扁平化数据
            seq_flat = seq_embs.reshape(-1, hidden_size)  # [B*L, H]
            pos_flat = pos_embs.reshape(-1, hidden_size)  # [B*L, H]
            loss_mask_flat = loss_mask.reshape(-1)        # [B*L]
            
            # 仅对有效位置计算
            valid_indices = torch.nonzero(loss_mask_flat, as_tuple=False).squeeze(-1)
            num_valid = valid_indices.numel()
            
            if num_valid == 0:
                return torch.tensor(0.0, device=seq_embs.device, requires_grad=True)
            
            # 准备点击/曝光掩码
            next_type_flat = None
            if next_action_type is not None:
                next_type_flat = next_action_type.reshape(-1)  # [B*L]
            
            # 样本级加权累计器
            sum_click_loss = torch.zeros((), device=seq_embs.device)
            sum_expo_loss = torch.zeros((), device=seq_embs.device)
            n_click = torch.zeros((), device=seq_embs.device)
            n_expo = torch.zeros((), device=seq_embs.device)

            # 统计日志用累计器（分块模式下也记录与非分块一致的标量）
            # 总体 pos/neg 平均值
            pos_sum_all = torch.zeros((), device=seq_embs.device)
            pos_count_all = torch.zeros((), device=seq_embs.device)
            neg_sum_all = torch.zeros((), device=seq_embs.device)
            neg_count_all = torch.zeros((), device=seq_embs.device)

            # 分类型（点击/曝光） pos/neg 平均值
            pos_sum_click = torch.zeros((), device=seq_embs.device)
            pos_count_click = torch.zeros((), device=seq_embs.device)
            neg_sum_click = torch.zeros((), device=seq_embs.device)
            neg_count_click = torch.zeros((), device=seq_embs.device)

            pos_sum_expo = torch.zeros((), device=seq_embs.device)
            pos_count_expo = torch.zeros((), device=seq_embs.device)
            neg_sum_expo = torch.zeros((), device=seq_embs.device)
            neg_count_expo = torch.zeros((), device=seq_embs.device)
            
            # 🚀 流式分块计算：逐块计算loss并累计
            for i in range(0, num_valid, chunk_size):
                idx_chunk = valid_indices[i: i + chunk_size]     # [C]
                q = seq_flat[idx_chunk]                          # [C, H]
                p = pos_flat[idx_chunk]                          # [C, H]
                
                # 正样本相似度（已normalize，cosine相似度等价于点积）
                raw_pos_logit = F.cosine_similarity(q, p, dim=-1)  # [C]
                # === [新增] 基于动作类型的 margin 调整（在温度缩放前进行） ===
                if getattr(self, 'enable_action_margin', False) and (next_type_flat is not None):
                    at = next_type_flat[idx_chunk]  # [C], 0=expo,1=click
                    margin_click = torch.as_tensor(self.action_margin_click, device=q.device, dtype=raw_pos_logit.dtype)
                    margin_expo = torch.as_tensor(self.action_margin_exposure, device=q.device, dtype=raw_pos_logit.dtype)
                    gamma = torch.where(at == 1, margin_click, margin_expo)  # [C]
                    raw_pos_logit = raw_pos_logit - gamma
                pos_logit = raw_pos_logit / self.temperature
                
                # === 🎯 负样本处理：使用融合后的负样本池 + O(1)自对比掩码 ===
                raw_neg_logits = torch.matmul(q, final_neg_embedding.transpose(-1, -2))  # [C, num_neg_total]
                
                # 🔧 O(1) 自对比掩码：使用预建的 index_map
                if inbatch_info is not None:
                    num_global_neg = inbatch_info['num_global_neg']
                    index_map = inbatch_info['index_map']  # [B*L]
                    rel_cols = index_map[idx_chunk]  # [C] 获取每个位置在valid_indices中的列号
                    
                    # 防御性检查：过滤潜在的-1（正常情况下不会出现）
                    valid_mask = rel_cols >= 0
                    if valid_mask.any():
                        self_cols = num_global_neg + rel_cols[valid_mask]  # 自身正样本在final_neg_embedding中的列
                        row_idx = torch.arange(idx_chunk.size(0), device=raw_neg_logits.device)[valid_mask]
                        raw_neg_logits[row_idx, self_cols] = float('-inf')
                
                neg_logits = raw_neg_logits / self.temperature  # 温度缩放用于loss
                
                # InfoNCE per-sample loss：-log(exp(pos) / (exp(pos) + sum(exp(neg))))
                # 等价于：log(exp(pos) + sum(exp(neg))) - pos，使用logsumexp提高数值稳定性
                lse_neg = torch.logsumexp(neg_logits, dim=-1)         # [C]
                lse_all = torch.logsumexp(torch.stack([pos_logit, lse_neg], dim=-1), dim=-1)
                loss_per_sample = lse_all - pos_logit                 # [C]
                
                # --- 日志统计：累计总体 pos/neg ---
                pos_sum_all += raw_pos_logit.sum()
                pos_count_all += raw_pos_logit.numel()
                neg_sum_all += raw_neg_logits[torch.isfinite(raw_neg_logits)].sum()
                neg_count_all += torch.isfinite(raw_neg_logits).sum()
                
                # 按点击/曝光分别累计
                if next_type_flat is None:
                    # 没有动作类型，全部当作点击处理
                    sum_click_loss += loss_per_sample.sum()
                    n_click += loss_per_sample.numel()
                else:
                    chunk_action_type = next_type_flat[idx_chunk]
                    click_mask_chunk = (chunk_action_type == 1)
                    expo_mask_chunk = (chunk_action_type == 0)
                    
                    if click_mask_chunk.any():
                        sum_click_loss += loss_per_sample[click_mask_chunk].sum()
                        n_click += click_mask_chunk.sum()
                        # 分类型日志：点击
                        pos_sum_click += raw_pos_logit[click_mask_chunk].sum()
                        pos_count_click += click_mask_chunk.sum()
                        neg_rows_click = raw_neg_logits[click_mask_chunk]  # 使用 raw_neg_logits
                        if neg_rows_click.numel() > 0:
                            neg_sum_click += neg_rows_click[torch.isfinite(neg_rows_click)].sum()
                            neg_count_click += torch.isfinite(neg_rows_click).sum()
                    
                    if expo_mask_chunk.any():
                        sum_expo_loss += loss_per_sample[expo_mask_chunk].sum()
                        n_expo += expo_mask_chunk.sum()
                        # 分类型日志：曝光
                        pos_sum_expo += raw_pos_logit[expo_mask_chunk].sum()
                        pos_count_expo += expo_mask_chunk.sum()
                        neg_rows_expo = raw_neg_logits[expo_mask_chunk]  # 使用 raw_neg_logits
                        if neg_rows_expo.numel() > 0:
                            neg_sum_expo += neg_rows_expo[torch.isfinite(neg_rows_expo)].sum()
                            neg_count_expo += torch.isfinite(neg_rows_expo).sum()
            
            # 样本级加权平均
            total_weighted_loss = sum_click_loss + exposure_weight * sum_expo_loss
            total_weight = n_click + exposure_weight * n_expo
            
            if total_weight > eps:
                loss = total_weighted_loss / total_weight
            else:
                loss = torch.tensor(0.0, device=seq_embs.device, requires_grad=True)
            
            # 计算平均损失用于返回
            avg_click_loss = (sum_click_loss / (n_click + eps)) if n_click > 0 else torch.tensor(0.0, device=seq_embs.device)
            avg_expo_loss = (sum_expo_loss / (n_expo + eps)) if n_expo > 0 else torch.tensor(0.0, device=seq_embs.device)
            
            # TensorBoard记录（分块模式：核心指标）
            if writer is not None and global_step is not None:
                # 🎯 In-batch 负样本统计（精简版）
                if self.enable_inbatch_negatives and global_step % 100 == 0:
                    inbatch_count = final_neg_embedding.shape[0] - clean_neg_embedding.shape[0]
                    writer.add_scalar("InBatch/inbatch_neg_count", inbatch_count, global_step)
                    writer.add_scalar("InBatch/total_neg_count", final_neg_embedding.shape[0], global_step)
                
                # 总体 pos/neg 平均相似度（未做温度缩放，与非分块模式一致）
                if pos_count_all.item() > 0:
                    writer.add_scalar("Model/nce_pos_logits", (pos_sum_all / (pos_count_all + 1e-5)).item(), global_step)
                if neg_count_all.item() > 0:
                    writer.add_scalar("Model/nce_neg_logits", (neg_sum_all / (neg_count_all + 1e-5)).item(), global_step)
                
                # 分类型loss（仅在有动作类型时记录）
                if next_action_type is not None:
                    if pos_count_click.item() > 0:
                        writer.add_scalar("Model/nce_pos_logits_click", (pos_sum_click / (pos_count_click + 1e-5)).item(), global_step)
                    if neg_count_click.item() > 0:
                        writer.add_scalar("Model/nce_neg_logits_click", (neg_sum_click / (neg_count_click + 1e-5)).item(), global_step)
                    if pos_count_expo.item() > 0:
                        writer.add_scalar("Model/nce_pos_logits_exposure", (pos_sum_expo / (pos_count_expo + 1e-5)).item(), global_step)
                    if neg_count_expo.item() > 0:
                        writer.add_scalar("Model/nce_neg_logits_exposure", (neg_sum_expo / (neg_count_expo + 1e-5)).item(), global_step)

                # 分类型loss均值
                if n_click.item() > 0:
                    writer.add_scalar("Model/nce_click_loss", (sum_click_loss / (n_click + 1e-5)).item(), global_step)
                if n_expo.item() > 0:
                    writer.add_scalar("Model/nce_exposure_loss", (sum_expo_loss / (n_expo + 1e-5)).item(), global_step)

        else:
            # 🔧 非分块模式：使用融合负样本池 + 自对比掩码
            # 1. 计算正样本相似度
            pos_logits = F.cosine_similarity(seq_embs, pos_embs, dim=-1).unsqueeze(-1) # [batch_size, maxlen, 1]
            # === [新增] 基于动作类型的 margin 调整（在温度缩放前进行） ===
            if getattr(self, 'enable_action_margin', False) and (next_action_type is not None):
                margin_click = torch.as_tensor(self.action_margin_click, device=pos_logits.device, dtype=pos_logits.dtype)
                margin_expo = torch.as_tensor(self.action_margin_exposure, device=pos_logits.device, dtype=pos_logits.dtype)
                gamma = torch.where(next_action_type == 1, margin_click, margin_expo).unsqueeze(-1)  # [B,L,1]
                pos_logits = pos_logits - gamma
            
            # 计算负样本相似度（使用融合后的负样本池）
            neg_logits = torch.matmul(seq_embs, final_neg_embedding.transpose(-1, -2)) # [batch_size, maxlen, num_total_neg]
            
            # 🔧 O(1) 自对比掩码：使用预建的 index_map
            if inbatch_info is not None:
                num_global_neg = inbatch_info['num_global_neg']
                index_map = inbatch_info['index_map']  # [B*L]
                valid_indices = inbatch_info['valid_indices']  # [num_valid]
                rel_cols = index_map[valid_indices]  # [num_valid] (0..num_valid-1)
                self_cols = num_global_neg + rel_cols  # [num_valid]
                
                # 展平后一次性设置掩码，避免多维索引开销
                flat_neg = neg_logits.reshape(-1, neg_logits.size(-1))
                flat_neg[valid_indices, self_cols] = float('-inf')
                neg_logits = flat_neg.view_as(neg_logits)
            
            # 拼接正负样本logits
            logits = torch.cat([pos_logits, neg_logits], dim=-1) # [batch_size, maxlen, 1+num_total_neg]
            
            # TensorBoard记录
            if writer is not None and global_step is not None:
                writer.add_scalar("Model/nce_pos_logits", pos_logits.mean().item(), global_step)
                writer.add_scalar("Model/nce_neg_logits", neg_logits[torch.isfinite(neg_logits)].mean().item(), global_step)
                
                # 🎯 In-batch 负样本统计（精简版）
                if self.enable_inbatch_negatives and global_step % 100 == 0:
                    inbatch_count = final_neg_embedding.shape[0] - clean_neg_embedding.shape[0]
                    writer.add_scalar("InBatch/inbatch_neg_count", inbatch_count, global_step)
                    writer.add_scalar("InBatch/total_neg_count", final_neg_embedding.shape[0], global_step)
                
                # 分别记录点击和曝光的相似度
                if next_action_type is not None:
                    click_mask = loss_mask & (next_action_type == 1)
                    exposure_mask = loss_mask & (next_action_type == 0)
                    
                    if click_mask.any():
                        click_pos_sim = pos_logits[click_mask.bool()].mean().item()
                        click_neg_logits = neg_logits[click_mask.bool()]
                        click_neg_sim = click_neg_logits[torch.isfinite(click_neg_logits)].mean().item()
                        writer.add_scalar("Model/nce_pos_logits_click", click_pos_sim, global_step)
                        writer.add_scalar("Model/nce_neg_logits_click", click_neg_sim, global_step)
                    
                    if exposure_mask.any():
                        exposure_pos_sim = pos_logits[exposure_mask.bool()].mean().item()
                        exposure_neg_logits = neg_logits[exposure_mask.bool()]
                        exposure_neg_sim = exposure_neg_logits[torch.isfinite(exposure_neg_logits)].mean().item()
                        writer.add_scalar("Model/nce_pos_logits_exposure", exposure_pos_sim, global_step)
                        writer.add_scalar("Model/nce_neg_logits_exposure", exposure_neg_sim, global_step)
            
            # 4. 支持曝光作为弱正样本的loss计算 - 修正为样本级加权
            if next_action_type is not None:
                # 区分点击和曝光样本
                click_mask = loss_mask & (next_action_type == 1)  # 点击样本
                exposure_mask = loss_mask & (next_action_type == 0)  # 曝光样本
                
                total_weighted_loss = torch.zeros((), device=logits.device)
                total_weight = torch.zeros((), device=logits.device)
                
                # 计算点击样本的loss（权重1.0）
                click_loss_total = torch.tensor(0.0, device=logits.device)
                expo_loss_total = torch.tensor(0.0, device=logits.device)
                
                if click_mask.any():
                    click_logits = logits[click_mask.bool()] / self.temperature
                    click_labels = torch.zeros(click_logits.size(0), device=logits.device, dtype=torch.int64)
                    click_loss_total = F.cross_entropy(click_logits, click_labels, reduction='sum')  # 改为sum
                    total_weighted_loss += click_loss_total
                    total_weight += click_mask.sum().float()
                    
                    if writer is not None and global_step is not None:
                        writer.add_scalar("Model/nce_click_loss", click_loss_total.item() / click_mask.sum().item(), global_step)
                
                # 计算曝光样本的loss（权重可配置，默认0.6）
                if exposure_mask.any():
                    exposure_logits = logits[exposure_mask.bool()] / self.temperature
                    exposure_labels = torch.zeros(exposure_logits.size(0), device=logits.device, dtype=torch.int64)
                    expo_loss_total = F.cross_entropy(exposure_logits, exposure_labels, reduction='sum')  # 改为sum
                    total_weighted_loss += exposure_weight * expo_loss_total  # 曝光样本权重
                    total_weight += exposure_weight * exposure_mask.sum().float()
                    
                    if writer is not None and global_step is not None:
                        writer.add_scalar("Model/nce_exposure_loss", expo_loss_total.item() / exposure_mask.sum().item(), global_step)
                
                if total_weight > eps:
                    loss = total_weighted_loss / total_weight
                else:
                    # 如果没有有效样本，返回0损失
                    loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
                
                # 计算平均损失
                avg_click_loss = (click_loss_total / (click_mask.sum().float() + eps)) if click_mask.any() else torch.tensor(0.0, device=logits.device)
                avg_expo_loss = (expo_loss_total / (exposure_mask.sum().float() + eps)) if exposure_mask.any() else torch.tensor(0.0, device=logits.device)
                
            else:
                # 原始logic：只使用loss_mask
                logits = logits[loss_mask.bool()] / self.temperature
                labels = torch.zeros(logits.size(0), device=logits.device, dtype=torch.int64)
                loss = F.cross_entropy(logits, labels)
                avg_click_loss = loss  # 没有区分时，全部当作click处理
                avg_expo_loss = torch.tensor(0.0, device=logits.device)
        
        if return_detailed_loss:
            return loss, avg_click_loss, avg_expo_loss
        else:
            return loss

