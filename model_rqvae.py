"""
重构的RQ-VAE模型 - 解决码本崩溃问题
主要改进：
1. 串行残差量化结构（真正的残差递进）
2. EMA码本更新机制替代K-Means初始化
3. 码本使用度监控和多样性损失
4. Dead Code重置机制
5. 数值稳定性优化

参考实现：
- Google "Residual Vector Quantization"
- DeepMind VQ-VAE 2
- Shopify VQ-Text
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from dataset import load_mm_emb
import warnings
from typing import List, Tuple, Optional, Dict

class MmEmbDataset(torch.utils.data.Dataset):
    """
    ⚡ 多进程优化版本：RQ-VAE训练数据集
    
    优化要点：
    1. 延迟tensor转换，避免多进程内存复制
    2. 在__getitem__中完成numpy到tensor的转换
    3. 保持数据处理在worker进程中进行，符合KISS原则
    4. 🔧 修复：正确使用OffsetMmEmbLoader

    Args:
        data_dir = os.environ.get('TRAIN_DATA_PATH')
        feature_id = MM emb ID
    """

    def __init__(self, data_dir, feature_id):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.feature_id = feature_id
        
        # 🔧 修复：使用OffsetMmEmbLoader加载多模态特征，RQ-VAE训练使用train数据
        self.mm_emb_loader = load_mm_emb(Path(data_dir, "creative_emb"), [feature_id], "train")
        
        # 🔧 修复：从loader的offsets中获取creative_id列表
        if hasattr(self.mm_emb_loader, 'offsets') and feature_id in self.mm_emb_loader.offsets:
            self.tid_list = list(self.mm_emb_loader.offsets[feature_id].keys())
            self.item_cnt = len(self.tid_list)
            print(f"✅ 加载特征 {feature_id}: {self.item_cnt} 个样本")
            
            # 检查样本数量
            if self.item_cnt == 0:
                raise ValueError(f"❌ 特征 {feature_id} 没有有效样本")
                
        else:
            available_features = list(self.mm_emb_loader.offsets.keys()) if hasattr(self.mm_emb_loader, 'offsets') else []
            raise ValueError(f"❌ 特征 {feature_id} 未找到，可用特征: {available_features}")
        
        # 预加载一个样本检查维度和数据有效性
        if self.item_cnt > 0:
            sample_emb = self.mm_emb_loader.get(feature_id, self.tid_list[0])
            if sample_emb is not None:
                # 确保是numpy数组
                if not isinstance(sample_emb, np.ndarray):
                    sample_emb = np.array(sample_emb, dtype=np.float32)
                    
                print(f"📊 特征 {feature_id} 维度: {sample_emb.shape}")
                
                # 验证维度是否与配置一致
                from config import get_rqvae_config
                try:
                    config = get_rqvae_config()[feature_id]
                    expected_dim = config["input_dim"]
                    actual_dim = sample_emb.shape[0] if len(sample_emb.shape) == 1 else sample_emb.shape[-1]
                    
                    if actual_dim != expected_dim:
                        print(f"⚠️ 警告: 特征 {feature_id} 维度不匹配 - 实际: {actual_dim}, 期望: {expected_dim}")
                    else:
                        print(f"✅ 特征 {feature_id} 维度验证通过")
                        
                except KeyError:
                    print(f"⚠️ 警告: 特征 {feature_id} 配置不存在")
                    
            else:
                print(f"⚠️ 警告: 特征 {feature_id} 第一个样本为空，将在训练时使用零向量")

    def __getitem__(self, index):
        """
        ⚡ 多进程优化：在worker进程中完成数据转换
        避免主进程预转换造成的内存复制问题
        🔧 修复：使用OffsetMmEmbLoader实时加载数据
        """
        tid = self.tid_list[index]
        
        # 🔧 修复：通过loader实时获取embedding
        emb_data = self.mm_emb_loader.get(self.feature_id, tid)
        
        if emb_data is None:
            # 如果数据为空，创建零向量
            from config import get_rqvae_config
            config = get_rqvae_config()[self.feature_id]
            input_dim = config["input_dim"]
            emb_data = np.zeros(input_dim, dtype=np.float32)
            print(f"⚠️ 警告: creative_id {tid} 的特征 {self.feature_id} 为空，使用零向量")
        
        # 确保是numpy数组
        if not isinstance(emb_data, np.ndarray):
            emb_data = np.array(emb_data, dtype=np.float32)
        
        # 转换为tensor
        tid_tensor = torch.tensor(tid, dtype=torch.long)
        emb_tensor = torch.tensor(emb_data, dtype=torch.float32)
        
        return tid_tensor, emb_tensor

    def __len__(self):
        return self.item_cnt

    @staticmethod
    def collate_fn(batch):
        tid, emb = zip(*batch)
        tid_batch, emb_batch = torch.stack(tid, dim=0), torch.stack(emb, dim=0)
        return tid_batch, emb_batch


class ResidualVectorQuantizer(torch.nn.Module):
    """
    改进的残差向量量化器 - 解决码本崩溃问题
    
    关键改进：
    1. EMA码本更新替代K-Means
    2. 码本使用度监控和重置
    3. 多样性损失和数值稳定性
    4. 真正的串行残差结构
    """
    
    def __init__(
        self,
        num_codebooks: int,
        codebook_size: int,
        codebook_dim: int,
        ema_decay: float = 0.99,
        epsilon: float = 1e-5,
        commitment_cost: float = 0.25,
        diversity_gamma: float = 0.1,
        dead_code_threshold: int = 100,
        device: str = 'cuda'
    ):
        super().__init__()
        
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.ema_decay = ema_decay
        self.epsilon = epsilon
        self.commitment_cost = commitment_cost
        self.diversity_gamma = diversity_gamma
        self.dead_code_threshold = dead_code_threshold
        self.device = device
        
        # 初始化码本embeddings
        self.codebooks = torch.nn.ModuleList([
            torch.nn.Embedding(codebook_size, codebook_dim)
            for _ in range(num_codebooks)
        ])
        
        # EMA统计量
        self.register_buffer('ema_cluster_size', torch.zeros(num_codebooks, codebook_size))
        self.register_buffer('ema_w', torch.zeros(num_codebooks, codebook_size, codebook_dim))
        
        # 码本使用统计
        self.register_buffer('codebook_usage', torch.zeros(num_codebooks, codebook_size))
        self.register_buffer('steps_since_last_use', torch.zeros(num_codebooks, codebook_size))
        self.register_buffer('total_steps', torch.tensor(0))
        
        # 初始化码本权重
        self._init_codebooks()
        
        print(f"🎯 ResidualVQ初始化: {num_codebooks}个码本, 每个{codebook_size}条目, 维度{codebook_dim}")
        print(f"📊 EMA衰减: {ema_decay}, 承诺损失: {commitment_cost}, 多样性损失: {diversity_gamma}")
    
    def _init_codebooks(self):
        """初始化码本权重 - 修正：按维度缩放而非码本大小"""
        import math
        # 使用标准的按维度缩放初始化，避免码本大小影响初始方差
        init_range = 1.0 / math.sqrt(self.codebook_dim)
        for codebook in self.codebooks:
            torch.nn.init.uniform_(codebook.weight, -init_range, init_range)
        
        # 初始化EMA统计量
        for i in range(self.num_codebooks):
            self.ema_w[i].copy_(self.codebooks[i].weight.data)
            self.ema_cluster_size[i].fill_(1.0)  # 避免除零
    
    def _compute_distances(self, z: torch.Tensor, codebook: torch.nn.Embedding) -> torch.Tensor:
        """计算输入与码本的L2距离"""
        # z: [batch_size, dim]
        # codebook.weight: [codebook_size, dim]
        
        z_flattened = z.view(-1, self.codebook_dim)
        
        # 计算平方L2距离: ||z - e||^2 = ||z||^2 + ||e||^2 - 2<z,e>
        z_norm_sq = torch.sum(z_flattened**2, dim=1, keepdim=True)  # [N, 1]
        e_norm_sq = torch.sum(codebook.weight**2, dim=1, keepdim=True).t()  # [1, K]
        
        distances = z_norm_sq + e_norm_sq - 2 * torch.matmul(z_flattened, codebook.weight.t())
        
        return distances
    
    def _update_ema(self, codebook_idx: int, encodings: torch.Tensor, z_e: torch.Tensor):
        """EMA更新码本"""
        if not self.training:
            return
            
        # encodings: [batch_size, codebook_size] one-hot
        # z_e: [batch_size, codebook_dim]
        
        with torch.no_grad():
            # 更新聚类大小统计
            cluster_size = encodings.sum(0)  # [codebook_size]
            self.ema_cluster_size[codebook_idx].mul_(self.ema_decay).add_(
                cluster_size, alpha=1 - self.ema_decay
            )
            
            # 更新权重统计
            dw = torch.matmul(encodings.t(), z_e)  # [codebook_size, codebook_dim]
            self.ema_w[codebook_idx].mul_(self.ema_decay).add_(dw, alpha=1 - self.ema_decay)
            
            # 更新码本权重
            n = self.ema_cluster_size[codebook_idx].sum()
            cluster_size_normalized = (
                (self.ema_cluster_size[codebook_idx] + self.epsilon) /
                (n + self.codebook_size * self.epsilon) * n
            )
            
            self.codebooks[codebook_idx].weight.data.copy_(
                self.ema_w[codebook_idx] / cluster_size_normalized.unsqueeze(1)
            )
    
    def _reset_dead_codes(self, codebook_idx: int, z_e: torch.Tensor):
        """重置死码"""
        if not self.training:
            return
            
        with torch.no_grad():
            # 找到长时间未使用的码
            dead_mask = self.steps_since_last_use[codebook_idx] > self.dead_code_threshold
            n_dead = dead_mask.sum().item()
            
            if n_dead > 0:
                # 随机选择一些活跃的样本来替换死码
                batch_size = z_e.shape[0]
                if batch_size > 0:
                    random_indices = torch.randperm(batch_size, device=self.device)[:n_dead]
                    dead_indices = torch.where(dead_mask)[0]
                    
                    # 用随机样本替换死码
                    self.codebooks[codebook_idx].weight.data[dead_indices] = z_e[random_indices]
                    
                    # 重置统计量
                    self.ema_cluster_size[codebook_idx][dead_indices] = 1.0
                    self.ema_w[codebook_idx][dead_indices] = z_e[random_indices]
                    self.steps_since_last_use[codebook_idx][dead_indices] = 0
                    
                    print(f"🔄 码本{codebook_idx}: 重置了{n_dead}个死码")
    
    def _compute_perplexity(self, encodings: torch.Tensor) -> torch.Tensor:
        """计算困惑度（衡量码本使用的均匀性）"""
        # encodings: [batch_size, codebook_size]
        avg_probs = encodings.mean(0)  # [codebook_size]
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return perplexity
    
    def quantize_layer(self, z: torch.Tensor, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """单层量化"""
        # z: [batch_size, codebook_dim]
        batch_size = z.shape[0]
        
        # 计算距离并找到最近的码本条目
        distances = self._compute_distances(z, self.codebooks[layer_idx])
        encoding_indices = torch.argmin(distances, dim=1)  # [batch_size]
        
        # 创建one-hot编码
        encodings = torch.zeros(batch_size, self.codebook_size, device=self.device)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)
        
        # 量化
        quantized = self.codebooks[layer_idx](encoding_indices)  # [batch_size, codebook_dim]
        
        # 更新统计信息
        if self.training:
            with torch.no_grad():
                # 更新使用统计
                unique_indices, counts = torch.unique(encoding_indices, return_counts=True)
                self.codebook_usage[layer_idx][unique_indices] += counts.float()
                
                # 更新步数统计
                self.steps_since_last_use[layer_idx] += 1
                self.steps_since_last_use[layer_idx][unique_indices] = 0
                
                # EMA更新
                self._update_ema(layer_idx, encodings, z)
                
                # 定期重置死码
                if self.total_steps % 1000 == 0:
                    self._reset_dead_codes(layer_idx, z)
        
        # 计算困惑度
        perplexity = self._compute_perplexity(encodings)
        
        # 计算损失组件
        commitment_loss = F.mse_loss(z, quantized.detach())
        codebook_loss = F.mse_loss(z.detach(), quantized)
        
        # Straight-through estimator
        quantized = z + (quantized - z).detach()
        
        stats = {
            'perplexity': perplexity,
            'commitment_loss': commitment_loss,
            'codebook_loss': codebook_loss,
            'usage_rate': (self.codebook_usage[layer_idx] > 0).float().mean()
        }
        
        return quantized, encoding_indices, encodings, stats
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], Dict]:
        """
        串行残差量化
        
        Args:
            z: 输入张量 [batch_size, codebook_dim]
            
        Returns:
            quantized: 最终量化结果
            indices_list: 每层的量化索引
            stats: 统计信息
        """
        batch_size = z.shape[0]
        residual = z
        quantized_sum = torch.zeros_like(z)
        
        indices_list = []
        all_stats = {}
        total_commitment_loss = 0.0
        total_codebook_loss = 0.0
        total_perplexity = 0.0
        total_usage_rate = 0.0
        
        # 串行残差量化
        for layer_idx in range(self.num_codebooks):
            # 量化当前残差
            quantized_layer, indices, encodings, stats = self.quantize_layer(residual, layer_idx)
            
            # 累加量化结果
            quantized_sum += quantized_layer
            
            # 计算新的残差
            residual = residual - quantized_layer
            
            # 保存索引
            indices_list.append(indices)
            
            # 累加统计信息
            total_commitment_loss += stats['commitment_loss']
            total_codebook_loss += stats['codebook_loss']
            total_perplexity += stats['perplexity']
            total_usage_rate += stats['usage_rate']
            
            # 保存每层统计
            all_stats[f'layer_{layer_idx}'] = stats
        
        # 计算多样性损失（鼓励码本均匀使用）
        diversity_loss = 0.0
        if self.training and self.diversity_gamma > 0:
            for layer_idx in range(self.num_codebooks):
                usage_prob = self.codebook_usage[layer_idx] / (self.codebook_usage[layer_idx].sum() + 1e-8)
                entropy = -torch.sum(usage_prob * torch.log(usage_prob + 1e-8))
                max_entropy = np.log(self.codebook_size)
                diversity_loss += self.diversity_gamma * (max_entropy - entropy)
        
        # 总损失
        vq_loss = (
            self.commitment_cost * total_commitment_loss +
            total_codebook_loss +
            diversity_loss
        )
        
        # 更新总步数
        if self.training:
            self.total_steps += 1
        
        # 汇总统计
        all_stats.update({
            'vq_loss': vq_loss,
            'commitment_loss': total_commitment_loss,
            'codebook_loss': total_codebook_loss,
            'diversity_loss': diversity_loss,
            'avg_perplexity': total_perplexity / self.num_codebooks,
            'avg_usage_rate': total_usage_rate / self.num_codebooks,
            'total_steps': self.total_steps.item()
        })
        
        return quantized_sum, indices_list, all_stats


class RQEncoder(torch.nn.Module):
    """
    改进的编码器 - 增加数值稳定性
    """
    def __init__(self, input_dim: int, hidden_channels: List[int], latent_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.layers = torch.nn.ModuleList()
        in_dim = input_dim
        
        for out_dim in hidden_channels:
            self.layers.append(torch.nn.Sequential(
                torch.nn.Linear(in_dim, out_dim),
                torch.nn.LayerNorm(out_dim),  # 添加LayerNorm提高稳定性
                torch.nn.GELU(),  # 使用GELU替代ReLU
                torch.nn.Dropout(dropout)
            ))
            in_dim = out_dim
        
        # 最后一层不加激活函数和dropout
        self.layers.append(torch.nn.Sequential(
            torch.nn.Linear(in_dim, latent_dim),
            torch.nn.LayerNorm(latent_dim)
        ))
        
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class RQDecoder(torch.nn.Module):
    """
    改进的解码器 - 增加数值稳定性
    """
    def __init__(self, latent_dim: int, hidden_channels: List[int], output_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.layers = torch.nn.ModuleList()
        in_dim = latent_dim
        
        for out_dim in reversed(hidden_channels):
            self.layers.append(torch.nn.Sequential(
                torch.nn.Linear(in_dim, out_dim),
                torch.nn.LayerNorm(out_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout)
            ))
            in_dim = out_dim
        
        # 最后一层：输出层
        self.layers.append(torch.nn.Linear(in_dim, output_dim))
        
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            z = layer(z)
        return z


class RQVAE(torch.nn.Module):
    """
    重构的RQ-VAE模型 - 解决码本崩溃问题
    
    主要改进：
    1. 串行残差量化结构
    2. EMA码本更新
    3. 码本健康监控
    4. 数值稳定性优化
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_channels: List[int],
        latent_dim: int,
        num_codebooks: int,
        codebook_size: int,
        ema_decay: float = 0.99,
        commitment_cost: float = 0.25,
        diversity_gamma: float = 0.1,
        encoder_dropout: float = 0.1,
        decoder_dropout: float = 0.1,
        device: str = 'cuda',
        **kwargs  # 兼容旧配置
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.device = device
        
        # 编码器
        self.encoder = RQEncoder(
            input_dim=input_dim,
            hidden_channels=hidden_channels,
            latent_dim=latent_dim,
            dropout=encoder_dropout
        ).to(device)
        
        # 解码器
        self.decoder = RQDecoder(
            latent_dim=latent_dim,
            hidden_channels=hidden_channels,
            output_dim=input_dim,
            dropout=decoder_dropout
        ).to(device)
        
        # 残差向量量化器
        self.quantizer = ResidualVectorQuantizer(
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            codebook_dim=latent_dim,
            ema_decay=ema_decay,
            commitment_cost=commitment_cost,
            diversity_gamma=diversity_gamma,
            device=device
        ).to(device)
        
        print(f"🎯 RQVAE模型初始化完成:")
        print(f"   输入维度: {input_dim} -> 潜在维度: {latent_dim}")
        print(f"   隐藏层: {hidden_channels}")
        print(f"   码本配置: {num_codebooks}个码本 × {codebook_size}条目")
        print(f"   EMA衰减: {ema_decay}, 承诺损失: {commitment_cost}")
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码"""
        return self.encoder(x)
    
    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """解码"""
        return self.decoder(z_q)
    
    def _get_codebook(self, x: torch.Tensor) -> torch.Tensor:
        """获取语义ID（用于推理）"""
        with torch.no_grad():
            z_e = self.encode(x)
            _, indices_list, _ = self.quantizer(z_e)
            # 将所有层的索引拼接
            semantic_ids = torch.stack(indices_list, dim=1)  # [batch_size, num_codebooks]
            return semantic_ids
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        前向传播
        
        Returns:
            x_hat: 重建结果
            semantic_ids: 语义ID
            recon_loss: 重建损失
            vq_loss: 量化损失
            total_loss: 总损失
            stats: 详细统计信息
        """
        # 编码
        z_e = self.encode(x)
        
        # 量化
        z_q, indices_list, stats = self.quantizer(z_e)
        
        # 解码
        x_hat = self.decode(z_q)
        
        # 计算重建损失
        recon_loss = F.mse_loss(x_hat, x, reduction='mean')
        
        # 获取量化损失
        vq_loss = stats['vq_loss']
        
        # 总损失
        total_loss = recon_loss + vq_loss
        
        # 语义ID
        semantic_ids = torch.stack(indices_list, dim=1)  # [batch_size, num_codebooks]
        
        # 更新统计信息
        stats.update({
            'recon_loss': recon_loss,
            'total_loss': total_loss
        })
        
        return x_hat, semantic_ids, recon_loss, vq_loss, total_loss, stats
    
    def get_codebook_stats(self) -> Dict:
        """获取码本健康状态"""
        stats = {}
        
        with torch.no_grad():
            for i in range(self.num_codebooks):
                usage = self.quantizer.codebook_usage[i]
                total_usage = usage.sum()
                
                if total_usage > 0:
                    usage_prob = usage / total_usage
                    entropy = -torch.sum(usage_prob * torch.log(usage_prob + 1e-8))
                    max_entropy = np.log(self.codebook_size)
                    
                    stats[f'codebook_{i}'] = {
                        'usage_rate': (usage > 0).float().mean().item(),
                        'entropy': entropy.item(),
                        'normalized_entropy': (entropy / max_entropy).item(),
                        'dead_codes': (self.quantizer.steps_since_last_use[i] > self.quantizer.dead_code_threshold).sum().item(),
                        'total_usage': total_usage.item()
                    }
                else:
                    stats[f'codebook_{i}'] = {
                        'usage_rate': 0.0,
                        'entropy': 0.0,
                        'normalized_entropy': 0.0,
                        'dead_codes': self.codebook_size,
                        'total_usage': 0
                    }
        
        return stats
    
    def reset_codebook_stats(self):
        """重置码本统计信息"""
        with torch.no_grad():
            self.quantizer.codebook_usage.zero_()
            self.quantizer.steps_since_last_use.zero_()
            self.quantizer.total_steps.zero_()
        print("🔄 码本统计信息已重置")


# 兼容性：保持旧的类名和接口
class RQ(ResidualVectorQuantizer):
    """兼容性包装"""
    def __init__(self, num_codebooks, codebook_size, codebook_emb_dim, shared_codebook, 
                 kmeans_method, kmeans_iters, distances_method, loss_beta, device):
        # 转换旧参数到新参数
        if isinstance(codebook_size, list):
            codebook_size = codebook_size[0]  # 使用第一个值
        
        super().__init__(
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_emb_dim,
            commitment_cost=loss_beta,
            device=device
        )
        warnings.warn("使用了旧的RQ接口，建议迁移到新的ResidualVectorQuantizer", DeprecationWarning)
    
    def forward(self, data):
        z_q, indices_list, stats = super().forward(data)
        # 返回旧格式
        vq_emb_list = [z_q]  # 简化返回
        semantic_id_list = torch.stack(indices_list, dim=1)
        rqvae_loss = stats['vq_loss']
        return vq_emb_list, semantic_id_list, rqvae_loss


# 兼容性：保持旧的VQEmbedding接口（废弃）
class VQEmbedding(torch.nn.Module):
    """废弃的VQEmbedding，建议使用新的ResidualVectorQuantizer"""
    def __init__(self, *args, **kwargs):
        super().__init__()
        warnings.warn("VQEmbedding已废弃，请使用ResidualVectorQuantizer", DeprecationWarning)
        raise NotImplementedError("VQEmbedding已废弃，请使用ResidualVectorQuantizer")