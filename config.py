"""
RQ-VAE配置文件 - 统一管理配置，避免重复代码
案例数据位于：/Users/helinfeng/Desktop/TencentGR_1k
"""
ratio = 1.5

def get_rqvae_config():
    """
    获取优化的RQVAE配置 - 解决码本崩溃问题，支持81-86特征
    
    主要改进：
    1. 增大码本尺寸，避免信息瓶颈
    2. 优化潜在维度，提供足够表达能力
    3. 使用EMA更新机制替代K-Means
    4. 添加多样性损失和健康监控
    5. 支持完整的81-86特征范围
    """
    
    # EMB_SHAPE_DICT定义各特征的输入维度
    EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    
    # 基础配置模板
    def get_feature_config(feature_id, input_dim):
        """根据特征ID和输入维度生成配置"""
        if input_dim <= 64:
            # 小维度特征配置（如81）
            return {
                "input_dim": input_dim,
                "hidden_channels": [min(input_dim * 2, 128), min(input_dim * 1.5, 96)],
                "latent_dim": max(input_dim // 2, 16),
                "num_codebooks": 2,
                "codebook_size": 128,
                "ema_decay": 0.99,
                "commitment_cost": 0.25,
                "diversity_gamma": 0.1,
                "encoder_dropout": 0.1,
                "decoder_dropout": 0.1,
            }
        elif input_dim <= 2048:
            # 中维度特征配置（如82）
            return {
                "input_dim": input_dim,
                "hidden_channels": [input_dim // 2, input_dim // 4, input_dim // 8],
                "latent_dim": max(input_dim // 16, 32),
                "num_codebooks": 3,
                "codebook_size": 256,
                "ema_decay": 0.99,
                "commitment_cost": 0.25,
                "diversity_gamma": 0.15,
                "encoder_dropout": 0.15,
                "decoder_dropout": 0.15,
            }
        else:
            # 高维度特征配置（如83-86）
            return {
                "input_dim": input_dim,
                "hidden_channels": [input_dim // 2, input_dim // 4, input_dim // 8, input_dim // 16],
                "latent_dim": max(input_dim // 32, 64),
                "num_codebooks": 4,
                "codebook_size": 512,
                "ema_decay": 0.99,
                "commitment_cost": 0.25,
                "diversity_gamma": 0.2,
                "encoder_dropout": 0.2,
                "decoder_dropout": 0.2,
            }
    
    # 生成所有特征的配置
    config = {}
    for feature_id, input_dim in EMB_SHAPE_DICT.items():
        base_config = get_feature_config(feature_id, input_dim)
        
        # 添加兼容性参数
        base_config.update({
            "shared_codebook": False,
            "kmeans_method": "ema",
            "kmeans_iters": 0,
            "distances_method": "l2",
            "loss_beta": base_config["commitment_cost"]  # 兼容性
        })
        
        config[feature_id] = base_config
    
    # 特殊优化：针对已知问题的特征进行微调；
    if "81" in config:
        config["81"].update({
            "latent_dim": 24,  # 手动优化
            "hidden_channels": [64, 48],
        })
    
    if "82" in config:
        config["82"].update({
            "latent_dim": 64,  # 手动优化  
            "hidden_channels": [512, 256, 128],
        })
    
    return config

def get_time_interval_config():
    """时间特征配置 - 统一的时间特征开关"""
    return {
        "time_gap_buckets": 32,              # 对数分桶数量（包含特殊的bucket 0）
        "max_time_gap": 86400 * 30,          # 最大时间间隔30天(秒)
        "min_time_gap": 1,                   # 最小时间间隔1秒
        "time_gap_embedding_dim": 16,        # 时间间隔embedding维度
        "enable_continuous_time_gap": True,  # 是否启用连续时间间隔特征
        "max_time_gap_norm": 72.0,          # 连续时间间隔归一化上界（小时）
        "enable_item_time_modulation": True, # 是否启用物品级时间调制（FiLM）
        "time_modulation_hidden_dim": 16,   # 时间调制MLP隐层维度
    }

def get_rope_config():
    """RoPE位置编码配置"""
    return {
        "enable_rope": True,  # 改为使用T5-style相对位置偏置
        "rope_theta": 10000.0,               # RoPE旋转基数
        "rope_max_seq_len": 128              # RoPE最大序列长度
    }

def get_embedding_config(args=None):
    """Embedding维度配置 - 自适应维度 + 启用分级投影策略"""
    # Field-wise投影功能已移除
    return {
        "ratio" : ratio,
        # 基础维度配置
        "id_embedding_dim": 64,              # ID类embedding维度(user_id, item_id)
        "sparse_embedding_dim": 64,          # 稀疏特征embedding维度(默认值，会被自适应覆盖) 
        "enable_adaptive_embedding": True,   # 启用自适应embedding维度
        
        # 自适应维度公式参数
        "adaptive_dim_formula": {
            "k": 8,                          # 比例因子
            "alpha": 0.25,                   # 指数因子(平方根的一半)
            "min_dim": 8,                    # 最小维度
            "max_dim": 96                    # 最大维度
        },
        
        # 时间特征专门配置
        "time_features_config": {
            "time_gap_embedding_dim": int(ratio * 8),    # 时间间隔embedding维度
            "action_type_embedding_dim": int(ratio * 8),  # 动作类型embedding维度
            "absolute_time_proj_dim": int(ratio * 32)     # 绝对时间投影维度（从64降至32）
        },
        
        # 🎯 连续特征投影配置
        "continual_features_config": {
            "user_continual_proj_dim": int(ratio * 8),   # 用户连续特征投影维度
            "item_continual_proj_dim": int(ratio * 8)    # 物品连续特征投影维度（更多特征，维度稍大）
        },
        
        # Field-wise投影配置已完全移除
        # 🎯 最终投影头配置 - 保证相似度空间对称
        "final_head": {
            "enabled": True,                         # 启用最终投影头
            "share_weights": False,                  # seq和cand不共享权重（保持灵活性）
            "l2_norm": True,                        # L2归一化
            "temperature_init": 0.04                # 温度参数初值
        },
        
        # 特定特征覆盖配置
        "per_feature_overrides": {
            # 🎯 维度统一到8的倍数，提升计算效率
            "111": {"out_dim": int(ratio * 48)},          # 从48降至40，vocab(4.7M)适度降维
            "121": {"out_dim": int(ratio * 48)},          # 从48降至40，vocab(2.1M)适度降维
            "115": {"out_dim": int(ratio * 16)},          # 从40降至16，vocab(691)用16维足够
            "102": {"out_dim": int(ratio * 40)},          # 保持40维，vocab(90k)
            "122": {"out_dim": int(ratio * 40)},          # 保持40维，vocab(90k)
            
            # 中等词表特征：优化维度对齐
            "118": {"out_dim": int(ratio * 24)},          # vocab(1.4k)保持24维
            "119": {"out_dim": int(ratio * 28)},          # 从28降至24，对齐8的倍数
            "120": {"out_dim": int(ratio * 28)},          # 从28降至24，对齐8的倍数
            "117": {"out_dim": int(ratio * 24)},          # 从24降至16，vocab(497)用16维足够
            
            # 小词表特征：精细调整
            "103": {"out_dim": int(ratio * 16)},          # vocab(86)保持16维
            "104": {"out_dim": int(ratio * 8)},           # vocab(2)保持8维
            "105": {"out_dim": int(ratio * 8)},           # vocab(7)保持8维
            "109": {"out_dim": int(ratio * 8)},           # vocab(3)保持8维
            "100": {"out_dim": int(ratio * 8)},           # vocab(6)保持8维
            "101": {"out_dim": int(ratio * 16)},          # 从12升至16，对齐常见维度
            "114": {"out_dim": int(ratio * 8)},           # vocab(20)保持8维
            "112": {"out_dim": int(ratio * 8)},           # vocab(30)保持8维
            "116": {"out_dim": int(ratio * 8)},           # vocab(18)保持8维
            "106": {"out_dim": int(ratio * 8)},           # 从12降至8，vocab(14)用8维足够
            "107": {"out_dim": int(ratio * 8)},           # 从12降至8，vocab(19)用8维足够
            "108": {"out_dim": int(ratio * 8)},           # vocab(4)保持8维
            "110": {"out_dim": int(ratio * 8)},           # vocab(2)保持8维
            
            # ID特征 - 优化user_id维度
            "item_id": {"out_dim": int(ratio * 64)}, 
            "user_id": {"out_dim": int(ratio * 48)}    # 从64降至32，user_id静态信息价值有限
        },
        
        # 投影层控制配置
        "projection_config": {
            "enable_individual_projections": False,  # 关闭逐特征投影
            "enable_unified_projection": True,       # 启用统一投影(concat后整体投影)
            "unified_projection_dim": None           # None表示使用hidden_units
        },
        
        # 调试配置
        "log_embedding_dim_decision": True         # 打印维度决策日志
    }

def get_semantic_id_config():
    """统一语义ID特征配置 - 改进为与RQ-VAE训练目标对齐的处理方式"""
    # 根据RQ-VAE配置自动生成对应的semantic_id特征配置
    rqvae_config = get_rqvae_config()
    # 动态生成所有特征的semantic_id配置
    semantic_id_features = {}
    for feature_id, config in rqvae_config.items():
        # 获取codebook配置
        num_codebooks = config["num_codebooks"]
        codebook_size = config["codebook_size"]
        latent_dim = config["latent_dim"]  # 🎯 新增：从RQ-VAE获取潜在维度
        
        # 确保codebook_size是整数
        if isinstance(codebook_size, list):
            max_codebook_size = max(codebook_size)
        else:
            max_codebook_size = codebook_size
        
        # 🎯 改进：使用RQ-VAE的潜在维度作为embedding维度，保持几何对齐
        embedding_dim = latent_dim  # 直接使用RQ-VAE的latent_dim
        
        # 统一格式：所有模式都使用连续数组格式
        semantic_id_features[feature_id] = {
            "feature_name": f"semantic_{feature_id}",
            "array_length": num_codebooks,  # 数组长度=codebook数量
            "vocab_size": max_codebook_size + 3,  # 有效范围 + 默认值 + padding值 + 保留值
            "embedding_dim": embedding_dim,  # 🎯 改进：使用RQ-VAE的latent_dim
            "default_value": max_codebook_size,      # 超出有效范围的默认值
            "padding_value": max_codebook_size + 1,   # padding值
            "feature_type": "array",  # 明确标记为数组特征
            
            "rqvae_aligned": True,  # 标记为与RQ-VAE对齐的处理方式
            "fusion_mode": "sum",   # 🎯 改进：默认使用等权求和（与RQ目标完全一致，无可学习权重）
            "enable_layer_weights": False,  # 🎯 改进：默认关闭可学习权重（sum模式下不需要）
            "reuse_codebook_weights": True,  # 优先复用RQ码本权重（仅端到端模式生效）
            "allow_fine_tune": True,  # 允许对复用权重进行微调
            "fine_tune_lr_ratio": 0.1,  # 微调学习率相对比例
            
            # 🎯 新增：其他融合模式的配置（可通过实验动态调整）
            "alternative_modes": {
                "weighted_sum": {
                    "enable_layer_weights": True,
                    "description": "使用可学习权重的加权求和"
                },
                "concat": {
                    "enable_layer_weights": False,
                    "description": "传统拼接模式，维度=embedding_dim*num_codebooks"
                },
                "hybrid": {
                    "enable_layer_weights": True,
                    "description": "sum+concat混合模式，需要额外投影层"
                }
            }
        }
    return {
        # semantic_id特征配置
        "semantic_id_features": semantic_id_features,
        "cache_file_pattern": "semantic_id_{feature_id}_{data_type}.pkl",
        "enable_semantic_id_validation": True,     # 是否在加载时验证semantic_id的有效性
        
        # 🎯 新增：全局RQ-VAE对齐配置
        "rqvae_alignment": {
            "enable_codebook_reuse": True,     # 🔥 启用全局开关：复用RQ码本权重（包括预计算模式）
            "enable_hybrid_fusion": False,    # 是否启用混合融合（sum + concat并联）
            "hybrid_fusion_weight": 0.7,     # 混合融合中sum分支的权重
            "enable_regularization": True,     # 🔥 启用对微调的码本权重的L2正则
            "regularization_weight": 1e-4,   # L2正则权重
            "enable_sid_dropout": False,       # 对SID embedding启用dropout
            "sid_dropout_rate": 0.1,          # SID dropout比例

            "enable_post_fusion_norm": True,  # 对融合后特征启用LayerNorm
            "post_fusion_norm_eps": 1e-5,     # LayerNorm的epsilon参数
            
            # 🎯 新增：预计算模式专用配置
            "enable_precompute_codebook_reuse": True,  # 预计算模式下是否复用codebook权重
            "precompute_fine_tune_lr_ratio": 0.1,      # 预计算模式下codebook微调学习率比例
        }
    }



def get_candidate_head_config():
    """
    🎯 候选侧头部类型配置 - 支持seq复杂/cand简单的架构优化
    
    Returns:
        dict: 候选侧头部配置选项
    """
    return {
        # 🔥 推荐配置：候选侧使用线性头，序列侧保持EnhancedDNN
        "linear": {
            "item_cand_head": "linear",
            "description": "候选侧使用nn.Linear，适合大规模检索和离线预计算",
            "benefits": ["训练稳定", "几何对齐好", "离线计算快", "避免过拟合"]
        },
        
        # 🔧 保守配置：维度对齐时使用Identity（零开销）
        "identity": {
            "item_cand_head": "identity", 
            "description": "维度对齐时使用nn.Identity，零计算开销",
            "benefits": ["零延迟", "完美对齐", "无参数", "最简架构"]
        },
        
        # 📊 传统配置：候选侧也使用EnhancedDNN（当前默认）
        "mlp": {
            "item_cand_head": "mlp",
            "description": "候选侧使用EnhancedDNN，表达力强但可能过拟合",
            "benefits": ["强表达力", "灵活性高", "传统方案"]
        },
        
        # 🎯 轻量配置：候选侧使用单层浅MLP
        "light_mlp": {
            "item_cand_head": "light_mlp",
            "description": "候选侧使用轻量MLP（无残差、低dropout）",
            "benefits": ["适度表达力", "较快计算", "中等复杂度"]
        }
    }
