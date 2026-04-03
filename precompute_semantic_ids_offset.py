#!/usr/bin/env python3
"""
预计算semantic_id脚本 - 使用训练好的RQ-VAE模型生成所有数据的semantic_id
将多模态特征转换为semantic_id并保存，供后续baseline模型快速加载使用

python precompute_semantic_ids.py --features 81 82 --data_type both
"""

import argparse
import json
import os
import pickle
from pathlib import Path
from tqdm import tqdm

import torch
import numpy as np

from model_rqvae import RQVAE
from config import get_rqvae_config, get_semantic_id_config
from dataset import OffsetMmEmbLoader

def _bulk_fetch_embeddings(mm_emb_loader: OffsetMmEmbLoader,
                           feature_id: str,
                           creative_ids: list,
                           emb_dim: int,
                           default_emb: np.ndarray) -> list:
    """
    批量按文件分组读取多模态向量，最小化I/O；返回与输入creative_ids顺序对齐的向量列表。
    - 81: 直接从内存字典读取
    - 82+: 使用偏移量一次性按文件读取多条
    """
    results: list = [None] * len(creative_ids)

    # 81 特征：offsets 保存的是字典 {creative_id: emb}
    if feature_id == '81':
        emb_map = mm_emb_loader.offsets.get('81', {})
        for idx, cid in enumerate(creative_ids):
            emb = emb_map.get(cid)
            if isinstance(emb, list):
                emb = np.asarray(emb, dtype=np.float32)
            if not isinstance(emb, np.ndarray) or emb is None or emb.shape[-1] != emb_dim:
                results[idx] = default_emb
            else:
                results[idx] = emb.astype(np.float32, copy=False)
        return results

    # 82+ 特征：offsets 保存 {creative_id: (file_path, offset)}
    mapping = mm_emb_loader.offsets.get(feature_id, {})
    groups = {}  # file_path -> list[(idx, cid, offset)]
    for idx, cid in enumerate(creative_ids):
        meta = mapping.get(cid)
        if meta is None:
            continue
        file_path, offset = meta
        groups.setdefault(file_path, []).append((idx, cid, offset))

    for file_path, items in groups.items():
        try:
            with open(file_path, 'rb') as f:
                for idx, cid, offset in items:
                    try:
                        f.seek(offset)
                        line = f.readline()
                        if not line:
                            results[idx] = default_emb
                            continue
                        data = json.loads(line.decode('utf-8').strip())
                        emb = data.get('emb')
                        if isinstance(emb, list):
                            emb = np.asarray(emb, dtype=np.float32)
                        if not isinstance(emb, np.ndarray) or emb.shape[-1] != emb_dim:
                            results[idx] = default_emb
                        else:
                            results[idx] = emb.astype(np.float32, copy=False)
                    except Exception:
                        results[idx] = default_emb
        except Exception:
            # 整个文件读取失败，文件内所有条目回退默认
            for idx, _, _ in items:
                results[idx] = default_emb

    # 对于未命中映射的，统一用默认
    for i in range(len(results)):
        if results[i] is None:
            results[i] = default_emb

    return results


def load_rqvae_model(feature_id, device='cuda'):
    """
    加载训练好的RQ-VAE模型
    
    Args:
        feature_id: 特征ID ('81' 或 '82')
        device: 设备
    
    Returns:
        RQVAE模型实例
    """
    cache_dir = os.environ.get('USER_CACHE_PATH')
    
    if not cache_dir:
        raise FileNotFoundError("USER_CACHE_PATH环境变量未设置，无法加载RQ-VAE模型")
    
    # 只从cache_dir查找模型
    model_paths = list(Path(cache_dir).glob(f"*rqvae_feat_{feature_id}_final/model.pt"))
    
    if not model_paths:
        raise FileNotFoundError(f"未找到特征 {feature_id} 的RQ-VAE模型文件，请检查USER_CACHE_PATH目录")
    
    # 选择最新的模型文件
    model_path = max(model_paths, key=lambda p: p.stat().st_mtime)
    print(f"📂 加载RQ-VAE模型: {model_path}")
    
    # 获取配置并初始化新的RQVAE模型
    config = get_rqvae_config()[feature_id]
    model = RQVAE(
        input_dim=config["input_dim"],
        hidden_channels=config["hidden_channels"],
        latent_dim=config["latent_dim"],
        num_codebooks=config["num_codebooks"],
        codebook_size=config["codebook_size"],
        ema_decay=config.get("ema_decay", 0.99),
        commitment_cost=config.get("commitment_cost", 0.25),
        diversity_gamma=config.get("diversity_gamma", 0.1),
        encoder_dropout=config.get("encoder_dropout", 0.1),
        decoder_dropout=config.get("decoder_dropout", 0.1),
        device=device
    ).to(device)
    
    # 加载权重
    model.load_state_dict(torch.load(model_path, map_location=device))

    # 🎯 新RQVAE模型结构：设置为推理模式
    # 新的RQVAE使用EMA更新机制，不需要手动设置codebook初始化状态
    model.eval()
    
    # 冻结所有参数，确保推理时不会意外更新
    for param in model.parameters():
        param.requires_grad = False
    
    print(f"✅ RQ-VAE模型加载完成: {feature_id}")
    return model


def check_codebook_health(semantic_ids_batch, feature_id):
    """
    检查码本使用的健康状态
    
    Args:
        semantic_ids_batch: 语义ID批次张量
        feature_id: 特征ID
    
    Returns:
        bool: 是否健康
    """
    unique_ids, counts = torch.unique(semantic_ids_batch, return_counts=True)
    
    # 获取配置中的码本大小
    config = get_rqvae_config()[feature_id]
    codebook_size = config['codebook_size']
    num_codebooks = config['num_codebooks']
    
    # 计算码本利用率
    total_possible_ids = codebook_size * num_codebooks
    utilization_rate = len(unique_ids) / total_possible_ids
    
    print(f"🔧 码本健康检查 - 特征{feature_id}:")
    print(f"   使用的不同ID数量: {len(unique_ids)}")
    print(f"   码本利用率: {utilization_rate:.1%}")
    # 安全打印ID范围（避免空张量报错）
    if unique_ids.numel() > 0:
        print(f"   ID分布: min={unique_ids.min().item()}, max={unique_ids.max().item()}")
    else:
        print(f"   ID分布: 空")
    
    # 检查是否存在码本崩塌
    is_healthy = True
    if len(unique_ids) < 10:  # 阈值可调
        print(f"⚠️ 码本崩塌风险：只使用了{len(unique_ids)}个不同ID")
        print(f"   使用的ID: {unique_ids.cpu().numpy().tolist()}")
        is_healthy = False
    
    if utilization_rate < 0.05:  # 利用率过低
        print(f"⚠️ 码本利用率过低：{utilization_rate:.1%}")
        is_healthy = False
    
    # 检查分布均匀性
    if len(counts) > 1:
        # 将counts转换为浮点并做零均值保护，避免不同PyTorch版本对整型std的限制
        counts_f = counts.to(torch.float32)
        mean = counts_f.mean()
        std = counts_f.std(unbiased=False)
        cv = (std / mean.clamp_min(1e-8)).item()
        if cv > 2.0:
            print(f"⚠️ 码本使用分布不均匀，变异系数: {cv:.2f}")
            is_healthy = False
    
    status = "✅ 健康" if is_healthy else "⚠️ 存在问题"
    print(f"   码本状态: {status}")
    
    return is_healthy


def validate_multimodal_features(mm_emb_loader, feature_id, sample_creative_ids):
    """
    验证多模态特征的质量
    
    Args:
        mm_emb_loader: 多模态特征加载器
        feature_id: 特征ID
        sample_creative_ids: 采样的creative_id列表
    
    Returns:
        bool: 数据质量是否合格
    """
    print(f"🔍 验证特征{feature_id}的多模态数据质量...")
    
    valid_count = 0
    zero_count = 0
    missing_count = 0
    norm_values = []
    
    sample_size = min(100, len(sample_creative_ids))
    for creative_id in sample_creative_ids[:sample_size]:
        emb_value = mm_emb_loader.get(feature_id, creative_id)
        if emb_value is None:
            missing_count += 1
        else:
            if not isinstance(emb_value, np.ndarray):
                emb_value = np.array(emb_value, dtype=np.float32)
            
            norm = np.linalg.norm(emb_value)
            norm_values.append(norm)
            
            if norm > 1e-6:
                valid_count += 1
            else:
                zero_count += 1
    
    print(f"📊 特征{feature_id}质量统计 (样本数={sample_size}):")
    print(f"   有效特征: {valid_count} ({valid_count/sample_size:.1%})")
    print(f"   零向量: {zero_count} ({zero_count/sample_size:.1%})")
    print(f"   缺失特征: {missing_count} ({missing_count/sample_size:.1%})")
    
    if norm_values:
        norm_array = np.array(norm_values)
        print(f"   向量范数统计: min={norm_array.min():.4f}, max={norm_array.max():.4f}, mean={norm_array.mean():.4f}")
    
    # 质量判断标准
    is_good_quality = (
        valid_count > zero_count and  # 有效特征应该多于零向量
        valid_count >= sample_size * 0.3  # 至少30%的样本有有效特征
    )
    
    status = "✅ 合格" if is_good_quality else "⚠️ 质量较差"
    print(f"   数据质量: {status}")
    
    return is_good_quality


def comprehensive_semantic_id_check(semantic_id_dict, feature_id):
    """
    全面的semantic ID健康检查
    
    Args:
        semantic_id_dict: semantic ID字典
        feature_id: 特征ID
    
    Returns:
        bool: 是否健康
    """
    print(f"\n🔍 特征{feature_id}的Semantic ID全面健康检查:")
    
    # 1. 统计semantic ID的分布
    all_semantic_ids = []
    for creative_id, semantic_array in semantic_id_dict.items():
        all_semantic_ids.extend(semantic_array)
    
    unique_ids, counts = np.unique(all_semantic_ids, return_counts=True)
    
    # 2. 检查码本利用率
    config = get_rqvae_config()[feature_id]
    codebook_size = config['codebook_size']
    num_codebooks = config['num_codebooks']
    total_possible_ids = codebook_size * num_codebooks
    utilization_rate = len(unique_ids) / total_possible_ids
    
    print(f"📊 码本利用率: {len(unique_ids)}/{total_possible_ids} = {utilization_rate:.1%}")
    if len(unique_ids) > 0:
        print(f"📊 ID范围: [{unique_ids.min()}, {unique_ids.max()}]")
        print(f"📊 最常用的10个ID: {unique_ids[np.argsort(counts)[-10:]].tolist()}")
        print(f"📊 使用频次分布: min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}, std={counts.std():.1f}")
    else:
        print(f"📊 当前无可用ID，跳过分布统计")
    
    # 3. 展示有多模态数据的creative_id及其semantic_id
    print(f"\n📋 Creative ID及其Semantic ID示例:")
    sample_ids = sorted(list(semantic_id_dict.keys()))[:10]
    for creative_id in sample_ids:
        semantic_array = semantic_id_dict[creative_id]
        print(f"   creative_id={creative_id}: {semantic_array}")
    
    # 4. 检查是否有配置化的默认值
    semantic_config = get_semantic_id_config()
    if feature_id in semantic_config['semantic_id_features']:
        default_value = semantic_config['semantic_id_features'][feature_id]['default_value']
        padding_value = semantic_config['semantic_id_features'][feature_id]['padding_value']
        
        default_count = np.sum(np.array(all_semantic_ids) == default_value)
        padding_count = np.sum(np.array(all_semantic_ids) == padding_value)
        
        print(f"\n📊 特殊值统计:")
        print(f"   默认值({default_value})使用次数: {default_count} ({default_count/len(all_semantic_ids):.1%})")
        print(f"   填充值({padding_value})使用次数: {padding_count} ({padding_count/len(all_semantic_ids):.1%})")
    
    # 5. 健康状态判断
    # 健康状态判断（为空时直接判定为不健康以提示问题）
    if len(unique_ids) == 0:
        is_healthy = False
    else:
        # 使用浮点计算变异系数并进行零均值保护
        counts_f = counts.astype(np.float32)
        cv = counts_f.std() / max(counts_f.mean(), 1e-8)
        is_healthy = (
            len(unique_ids) >= 20 and
            utilization_rate >= 0.05 and
            cv < 3.0
        )
    
    status = "✅ 健康" if is_healthy else "⚠️ 存在问题"
    print(f"\n🎯 码本健康状态: {status}")
    
    if not is_healthy:
        print(f"💡 建议:")
        if len(unique_ids) < 20:
            print(f"   - 码本多样性不足，考虑增加训练数据或调整模型参数")
        if utilization_rate < 0.05:
            print(f"   - 码本利用率过低，可能存在码本崩塌")
        if counts.std() / counts.mean() >= 3.0:
            print(f"   - 码本使用分布不均匀，部分码本过度使用")
    
    return is_healthy


def collect_creative_ids_from_data(data_paths, data_type="train"):
    """
    从数据文件中收集creative_id，并加载indexer进行ID映射
    
    Args:
        data_paths: 数据文件路径列表
        data_type: 数据类型 ("train" 或 "eval")
    
    Returns:
        tuple: (original_creative_ids_set, indexer_dict)
            - original_creative_ids_set: 原始creative_id的集合（字符串格式）
            - indexer_dict: indexer字典，用于后续映射
    """
    # 🔧 加载indexer进行ID映射
    if data_type == "train":
        data_dir = os.environ.get('TRAIN_DATA_PATH')
    else:  # eval
        data_dir = os.environ.get('EVAL_DATA_PATH')
    
    if not data_dir:
        raise ValueError(f"{data_type.upper()}_DATA_PATH环境变量未设置")
    
    indexer_path = Path(data_dir) / 'indexer.pkl'
    if not indexer_path.exists():
        raise FileNotFoundError(f"indexer文件不存在: {indexer_path}")
    
    # 加载indexer
    with open(indexer_path, 'rb') as f:
        indexer = pickle.load(f)
    
    print(f"📂 已加载indexer: {indexer_path}")
    print(f"   物品数量: {len(indexer['i'])}")
    
    # 构建反向映射：reid -> original_creative_id
    indexer_i_rev = {v: k for k, v in indexer['i'].items()}
    
    creative_ids_reid = set()  # 收集reid (整数)
    
    for data_path in data_paths:
        if not os.path.exists(data_path):
            print(f"⚠️  数据文件不存在，跳过: {data_path}")
            continue
            
        print(f"📂 收集creative_id ({data_type}): {data_path}")
        with open(data_path, 'r') as f:
            for line in tqdm(f, desc=f"扫描{Path(data_path).name}"):
                data = json.loads(line)
                
                # 数据格式：每行是一个用户的完整序列，包含多个交互记录
                # 每个记录格式：[user_id, item_id, user_feat, item_feat, action_type, timestamp]
                if isinstance(data, list):
                    # 用户序列数据：遍历所有交互记录
                    for record in data:
                        if isinstance(record, list) and len(record) >= 2:
                            # record[1] 是 item_id (这里应该是reid)
                            item_reid = record[1]
                            if item_reid and item_reid != 0:  # 跳过空值和padding
                                creative_ids_reid.add(item_reid)
                
                elif isinstance(data, dict):
                    # 字典格式：检查常见的字段名
                    if 'seq' in data:
                        # 序列格式
                        for item_reid in data['seq']:
                            if item_reid != 0:  # 跳过padding
                                creative_ids_reid.add(item_reid)
                    
                    if 'creative_id' in data:
                        # 如果直接是creative_id，可能需要转换
                        creative_id = data['creative_id']
                        if isinstance(creative_id, str):
                            # 字符串格式，查找对应的reid
                            if creative_id in indexer['i']:
                                creative_ids_reid.add(indexer['i'][creative_id])
                        else:
                            # 整数格式，直接添加
                            creative_ids_reid.add(creative_id)
                    
                    if 'item_id' in data:
                        # 单个item格式（另一种命名）
                        item_id = data['item_id']
                        if isinstance(item_id, str):
                            # 字符串格式，查找对应的reid
                            if item_id in indexer['i']:
                                creative_ids_reid.add(indexer['i'][item_id])
                        else:
                            # 整数格式，直接添加
                            creative_ids_reid.add(item_id)
    
    print(f"📊 {data_type}数据收集到 {len(creative_ids_reid)} 个唯一reid")
    
    # 🔧 将reid转换为原始creative_id（字符串）
    original_creative_ids = set()
    for item_reid in creative_ids_reid:
        if item_reid in indexer_i_rev:
            original_creative_id = indexer_i_rev[item_reid]
            original_creative_ids.add(original_creative_id)
        else:
            print(f"⚠️  未找到reid {item_reid} 对应的原始creative_id")
    
    print(f"📊 转换后的原始creative_id数量: {len(original_creative_ids)}")
    print(f"📋 原始creative_id示例: {list(original_creative_ids)[:10]}")
    
    return original_creative_ids, indexer


def precompute_semantic_ids_for_feature(feature_id, creative_ids, data_type="train", mm_emb_path=None, device='cuda', batch_size=1024, use_training_strategy=True):
    """
    为指定特征预计算所有creative_id的semantic_id
    
    Args:
        feature_id: 特征ID
        creative_ids: 要处理的creative_id集合
        data_type: 数据类型 ("train" 或 "eval")
        device: 设备
        batch_size: 批次大小
        use_training_strategy: 是否使用与训练时相同的策略（推荐）
    
    Returns:
        dict: {creative_id: semantic_id_list}
    """
    print(f"\n🎯 开始预计算特征 {feature_id} 的semantic_id ({data_type})...")
    
    # 加载RQ-VAE模型
    rqvae_model = load_rqvae_model(feature_id, device)
    
    # 🔧 修复：使用传入的多模态数据路径，与dataset.py保持一致
    if mm_emb_path is None:
        # 兼容旧版本调用方式
        if data_type == "train":
            data_path = os.environ.get('TRAIN_DATA_PATH')
        else:  # eval
            data_path = os.environ.get('EVAL_DATA_PATH')
        
        if not data_path or not os.path.exists(data_path):
            raise RuntimeError(f"数据路径不存在: {data_path}")
        mm_emb_path = Path(data_path, "creative_emb")
    
    # 确保mm_emb_path是Path对象
    if isinstance(mm_emb_path, str):
        mm_emb_path = Path(mm_emb_path)
    
    if not mm_emb_path.exists():
        raise RuntimeError(f"多模态数据路径不存在: {mm_emb_path}")
    
    # 🔧 使用Offset方式加载多模态数据（按需读取，避免整量入内存）
    print(f"📂 使用偏移量加载多模态数据: {mm_emb_path} (data_type: {data_type})")
    try:
        mm_emb_loader = OffsetMmEmbLoader(mm_emb_path, [feature_id], data_type)
        print(f"✅ 偏移量加载器就绪: {mm_emb_path}")
    except Exception as e:
        print(f"❌ 加载偏移量失败: {e}")
        raise RuntimeError(f"无法加载偏移量或特征数据，请检查 {mm_emb_path}")
    
    # 🔧 修复：使用config.py中的统一配置，避免硬编码
    from config import get_rqvae_config
    rqvae_config = get_rqvae_config()
    
    # 从RQ-VAE配置中获取embedding维度
    if feature_id in rqvae_config:
        emb_dim = rqvae_config[feature_id]["input_dim"]
    else:
        # 兼容性：如果配置中没有该特征，使用默认维度映射
        EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
        emb_dim = EMB_SHAPE_DICT.get(feature_id, 1024)
        print(f"⚠️  特征 {feature_id} 未在RQ-VAE配置中找到，使用默认维度: {emb_dim}")
    
    default_emb = np.zeros(emb_dim, dtype=np.float32)
    
    # 🔍 检查多模态特征覆盖率，但不过滤（保持完整性）
    available_creative_ids = set(mm_emb_loader.offsets.get(feature_id, {}).keys())
    valid_creative_ids = set(creative_ids) & available_creative_ids
    
    print(f"📊 数据覆盖率分析:")
    print(f"   需要处理的creative_id数量: {len(creative_ids)}")
    print(f"   有多模态特征的数量: {len(available_creative_ids)}")
    print(f"   有效覆盖数量: {len(valid_creative_ids)}")
    if len(creative_ids) > 0:
        print(f"   覆盖率: {len(valid_creative_ids)/len(creative_ids):.1%}")
    print(f"   缺失数量: {len(creative_ids) - len(valid_creative_ids)}")
    
    # 🔍 详细诊断信息
    if len(valid_creative_ids) == 0:
        print(f"❌ 没有找到任何有效的creative_id！")
        print(f"🔍 诊断信息:")
        print(f"   - 输入creative_ids前5个: {list(creative_ids)[:5]}")
        print(f"   - 可用creative_ids前5个: {list(available_creative_ids)[:5]}")
        print(f"   - 多模态数据路径: {mm_emb_path}")
        print(f"   - 特征ID: {feature_id}")
        
        # 检查数据文件是否存在
        if feature_id == '81':
            pkl_file = mm_emb_path / f'emb_{feature_id}_32.pkl'
            print(f"   - 81特征文件存在: {pkl_file.exists()}")
        else:
            shape = {"82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}.get(feature_id, 1024)
            json_dir = mm_emb_path / f'emb_{feature_id}_{shape}'
            print(f"   - JSON目录存在: {json_dir.exists()}")
            if json_dir.exists():
                json_files = list(json_dir.glob('*.json'))
                print(f"   - JSON文件数量: {len(json_files)}")
        
        # 不再抛出错误，而是继续处理（全部使用默认值）
        print(f"⚠️ 将为所有creative_id使用默认语义ID")
    
    # 🔍 数据质量预检
    if valid_creative_ids:
        sample_creative_ids = sorted(list(valid_creative_ids))
        data_quality_ok = validate_multimodal_features(mm_emb_loader, feature_id, sample_creative_ids)
        if not data_quality_ok:
            print(f"⚠️ 数据质量检查未通过，但继续处理...")
    
    # 🎯 优先展示有多模态数据的creative_id示例
    if valid_creative_ids:
        valid_examples = sorted(list(valid_creative_ids))[:20]  # 取前20个
        print(f"📋 有多模态特征的creative_id示例: {valid_examples}")
        
        # 展示几个具体样本的特征值
        print(f"🔍 多模态特征样本检查:")
        for i, creative_id in enumerate(valid_examples[:5]):  # 只检查前5个
            emb_value = mm_emb_loader.get(feature_id, creative_id)
            if emb_value is not None:
                if not isinstance(emb_value, np.ndarray):
                    emb_value = np.array(emb_value, dtype=np.float32)
                norm = np.linalg.norm(emb_value)
                print(f"   creative_id={creative_id}: 维度={emb_value.shape}, 范数={norm:.4f}, 范围=[{emb_value.min():.4f}, {emb_value.max():.4f}]")
            else:
                print(f"   creative_id={creative_id}: 特征为空")
    else:
        print(f"⚠️ 没有找到任何有效的多模态特征！")
    
    # 展示缺失多模态数据的creative_id示例
    missing_creative_ids = set(creative_ids) - valid_creative_ids
    if missing_creative_ids:
        missing_examples = sorted(list(missing_creative_ids))[:10]  # 取前10个
        print(f"📋 缺失多模态特征的creative_id示例: {missing_examples}")
    
    # 🔧 移除错误抛出，允许程序继续处理（全部使用默认值）
    # if len(valid_creative_ids) == 0:
    #     raise RuntimeError(f"没有找到任何有效的creative_id，请检查数据路径和特征ID")
    
    semantic_id_dict = {}
    
    if use_training_strategy:
        # 🎯 策略1：使用训练时相同的策略 - 分别处理有效和缺失的creative_id
        print(f"🎯 使用训练时相同的策略：优先处理有效特征，缺失特征使用配置化默认值")
        
        # 第一步：为有多模态特征的creative_id生成语义ID
        valid_creative_ids_list = list(valid_creative_ids)
        print(f"📊 第一步：处理 {len(valid_creative_ids_list)} 个有效creative_id...")
        
        # 🔧 添加空集保护：避免在空集合上调用min/max
        if len(valid_creative_ids_list) > 0:
            print(f"📋 有效creative_id范围: [{min(valid_creative_ids_list)}, {max(valid_creative_ids_list)}]")
            print(f"📋 有效creative_id示例: {valid_creative_ids_list[:10]}")
        else:
            print(f"📋 没有有效的creative_id，跳过RQ-VAE编码步骤")
        
        with torch.no_grad():
            for start_idx in tqdm(range(0, len(valid_creative_ids_list), batch_size), 
                                 desc=f"处理有效特征_{feature_id}"):
                end_idx = min(start_idx + batch_size, len(valid_creative_ids_list))
                batch_creative_ids = valid_creative_ids_list[start_idx:end_idx]
                
                # 批量按文件分组读取多模态特征（最小化I/O）
                batch_embeddings = _bulk_fetch_embeddings(
                    mm_emb_loader=mm_emb_loader,
                    feature_id=feature_id,
                    creative_ids=batch_creative_ids,
                    emb_dim=emb_dim,
                    default_emb=default_emb,
                )
                
                # 转换为tensor并生成语义ID
                batch_tensor = torch.from_numpy(np.stack(batch_embeddings, axis=0)).to(device)
                _, semantic_ids_batch, _, _, _, _ = rqvae_model(batch_tensor)
                
                # 🔍 在第一个批次进行码本健康检查
                if start_idx == 0:
                    codebook_health = check_codebook_health(semantic_ids_batch, feature_id)
                    if not codebook_health:
                        print(f"⚠️ 码本健康检查未通过，但继续处理...")
                
                # 保存结果
                for i, creative_id in enumerate(batch_creative_ids):
                    sample_semantic_ids = semantic_ids_batch[i].cpu().numpy()
                    semantic_array = [int(x) for x in sample_semantic_ids.flatten()]
                    
                    # 确保数组长度与配置一致
                    from config import get_semantic_id_config
                    semantic_config = get_semantic_id_config()
                    if feature_id in semantic_config['semantic_id_features']:
                        expected_length = semantic_config['semantic_id_features'][feature_id]['array_length']
                        if len(semantic_array) < expected_length:
                            padding_value = semantic_config['semantic_id_features'][feature_id]['padding_value']
                            semantic_array.extend([padding_value] * (expected_length - len(semantic_array)))
                        elif len(semantic_array) > expected_length:
                            semantic_array = semantic_array[:expected_length]
                    
                    semantic_id_dict[creative_id] = semantic_array
        
        # 第二步：为缺失多模态特征的creative_id分配配置化的默认语义ID
        missing_creative_ids = set(creative_ids) - valid_creative_ids
        if missing_creative_ids:
            print(f"📊 第二步：为 {len(missing_creative_ids)} 个缺失特征的creative_id分配默认语义ID...")
            
            # 🔧 添加空集保护：避免在空集合上调用min/max
            missing_creative_ids_list = list(missing_creative_ids)
            if len(missing_creative_ids_list) > 0:
                print(f"📋 缺失creative_id范围: [{min(missing_creative_ids_list)}, {max(missing_creative_ids_list)}]")
                print(f"📋 缺失creative_id示例: {sorted(missing_creative_ids_list)[:10]}")
            
            from config import get_semantic_id_config
            semantic_config = get_semantic_id_config()
            if feature_id in semantic_config['semantic_id_features']:
                feature_config = semantic_config['semantic_id_features'][feature_id]
                default_array = [feature_config['default_value']] * feature_config['array_length']
                
                for creative_id in missing_creative_ids:
                    semantic_id_dict[creative_id] = default_array.copy()
                
                print(f"✅ 为缺失特征分配默认语义ID: {default_array}")
                print(f"📊 缺失特征处理完成: {len(missing_creative_ids)} 个creative_id")
        else:
            print(f"✅ 所有creative_id都有多模态特征，无需处理缺失特征")
    
    else:
        # 🎯 策略2：已弃用 - 统一使用配置化默认值策略以确保一致性
        print(f"⚠️ 策略2已弃用，自动切换到推荐的配置化默认值策略")
        
        # 为缺失多模态特征的creative_id分配配置化的默认语义ID
        missing_creative_ids = set(creative_ids) - valid_creative_ids
        if missing_creative_ids:
            print(f"📊 为 {len(missing_creative_ids)} 个缺失特征的creative_id分配默认语义ID...")
            
            # 🔧 添加空集保护：避免在空集合上调用min/max
            missing_creative_ids_list = list(missing_creative_ids)
            if len(missing_creative_ids_list) > 0:
                print(f"📋 缺失creative_id范围: [{min(missing_creative_ids_list)}, {max(missing_creative_ids_list)}]")
                print(f"📋 缺失creative_id示例: {sorted(missing_creative_ids_list)[:10]}")
            
            semantic_config = get_semantic_id_config()
            if feature_id in semantic_config['semantic_id_features']:
                feature_config = semantic_config['semantic_id_features'][feature_id]
                default_array = [feature_config['default_value']] * feature_config['array_length']
                
                for creative_id in missing_creative_ids:
                    semantic_id_dict[creative_id] = default_array.copy()
                
                print(f"✅ 为缺失特征分配默认语义ID: {default_array}")
                print(f"📊 缺失特征处理完成: {len(missing_creative_ids)} 个creative_id")
        else:
            print(f"✅ 所有creative_id都有多模态特征，无需处理缺失特征")
    
    print(f"✅ 特征 {feature_id} 的semantic_id预计算完成: {len(semantic_id_dict)} 个items")
    
    # 🔍 进行全面的semantic ID健康检查
    print(f"\n🔍 开始全面健康检查...")
    is_healthy = comprehensive_semantic_id_check(semantic_id_dict, feature_id)
    
    if is_healthy:
        print(f"🎉 特征{feature_id}的semantic ID生成成功且健康！")
    else:
        print(f"⚠️ 特征{feature_id}的semantic ID存在潜在问题，请检查上述建议")
    
    return semantic_id_dict


def save_semantic_id_dict(semantic_id_dict, feature_id, data_type="train"):
    """
    保存semantic_id字典到缓存目录
    
    Args:
        semantic_id_dict: semantic_id字典
        feature_id: 特征ID
        data_type: 数据类型 ("train" 或 "eval")
    """
    cache_dir = os.environ.get('USER_CACHE_PATH')
    if not cache_dir:
        raise ValueError("USER_CACHE_PATH环境变量未设置，无法保存semantic_id数据")
    
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # 分离训练和推理的缓存文件
    if data_type == "train":
        semantic_file = cache_path / f"semantic_id_{feature_id}_train.pkl"
        json_file = cache_path / f"semantic_id_{feature_id}_train.json"
    else:  # eval
        semantic_file = cache_path / f"semantic_id_{feature_id}_eval.pkl"
        json_file = cache_path / f"semantic_id_{feature_id}_eval.json"
    
    # 保存为pickle文件
    with open(semantic_file, 'wb') as f:
        pickle.dump(semantic_id_dict, f)
    
    # 保存为json文件（备份，便于调试）
    with open(json_file, 'w') as f:
        json.dump(semantic_id_dict, f, indent=2)
    
    print(f"💾 semantic_id字典已保存 ({data_type}):")
    print(f"   - 主文件: {semantic_file}")
    print(f"   - 备份文件: {json_file}")
    
    # 输出统计信息
    total_items = len(semantic_id_dict)
    sample_items = list(semantic_id_dict.items())[:50]
    print(f"📊 保存统计: {total_items} 个items")
    print(f"📋 示例数据:")
    for creative_id, semantic_ids in sample_items:
        print(f"   creative_id={creative_id} -> semantic_ids={semantic_ids[:50]}...")


def main():
    parser = argparse.ArgumentParser(description='预计算semantic_id脚本')
    parser.add_argument('--features', nargs='+', default=['81'], 
                       choices=['81', '82', '83', '84', '85', '86'], 
                       help='要预计算的特征ID')
    parser.add_argument('--device', default='cuda', help='计算设备')
    parser.add_argument('--batch_size', type=int, default=1024, help='批次大小')
    parser.add_argument('--force_recompute', action='store_true',default=True, 
                       help='强制重新计算，即使缓存文件已存在')
    parser.add_argument('--data_type', choices=['train', 'eval', 'both'], default='both',
                       help='要预计算的数据类型：train（训练数据）、eval（推理数据）、both（两者都计算）')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🎯 semantic_id预计算开始...")
    print("="*60)
    print(f"特征列表: {args.features}")
    print(f"设备: {args.device}")
    print(f"批次大小: {args.batch_size}")
    print(f"数据类型: {args.data_type}")
    print(f"强制重新计算: {args.force_recompute}")
    
    # 确定要处理的数据类型
    data_types_to_process = []
    if args.data_type == 'both':
        data_types_to_process = ['train', 'eval']
    else:
        data_types_to_process = [args.data_type]
    
    # 为每个数据类型和特征预计算semantic_id
    for data_type in data_types_to_process:
        print(f"\n{'='*60}")
        print(f"处理 {data_type.upper()} 数据")
        print(f"{'='*60}")
        
        # 获取对应的数据文件路径
        if data_type == 'train':
            data_path = os.environ.get('TRAIN_DATA_PATH')
            if not data_path:
                print(f"❌ TRAIN_DATA_PATH环境变量未设置，跳过训练数据")
                continue
            print(f"📁 训练数据根目录: {data_path}")
            print(f"📁 目录是否存在: {os.path.exists(data_path)}")
            # 多模态数据路径应该是 data_path/creative_emb
            mm_emb_path = Path(data_path, "creative_emb")
            print(f"📁 多模态数据路径: {mm_emb_path}")
            print(f"📁 多模态数据是否存在: {mm_emb_path.exists()}")
            # 训练数据文件是 seq.jsonl
            data_files = [
                os.path.join(data_path, 'seq.jsonl')
            ]
            print(f"📁 训练数据文件: {data_files[0]}")
            print(f"📁 训练数据文件是否存在: {os.path.exists(data_files[0])}")
        else:  # eval
            data_path = os.environ.get('EVAL_DATA_PATH')
            if not data_path:
                print(f"❌ EVAL_DATA_PATH环境变量未设置，跳过推理数据")
                continue
            print(f"📁 推理数据根目录: {data_path}")
            print(f"📁 目录是否存在: {os.path.exists(data_path)}")
            # 多模态数据路径应该是 data_path/creative_emb
            mm_emb_path = Path(data_path, "creative_emb")
            print(f"📁 多模态数据路径: {mm_emb_path}")
            print(f"📁 多模态数据是否存在: {mm_emb_path.exists()}")
            # 推理数据文件是 predict_seq.jsonl
            data_files = [
                os.path.join(data_path, 'predict_seq.jsonl')
            ]
            print(f"📁 推理数据文件: {data_files[0]}")
            print(f"📁 推理数据文件是否存在: {os.path.exists(data_files[0])}")
        
        # 收集creative_id
        print(f"📂 扫描{data_type}数据文件，收集creative_id...")
        creative_ids, indexer = collect_creative_ids_from_data(data_files, data_type)
        
        if not creative_ids:
            print(f"❌ {data_type}数据中未找到任何creative_id，跳过")
            print(f"🔧 请检查数据格式是否正确")
            print(f"🔧 数据文件路径: {data_files}")
            continue
        
        print(f"📊 {data_type}数据creative_id数量: {len(creative_ids)}")
        print(f"📊 示例creative_id: {list(creative_ids)[:10]}")  # 减少展示数量
        
        # 为每个特征预计算semantic_id
        for feature_id in args.features:
            # 检查是否已存在缓存文件
            cache_dir = os.environ.get('USER_CACHE_PATH')
            if not cache_dir:
                raise ValueError("USER_CACHE_PATH环境变量未设置，无法检查缓存文件")
            cache_file = Path(cache_dir) / f"semantic_id_{feature_id}_{data_type}.pkl"
            
            if cache_file.exists() and not args.force_recompute:
                print(f"📋 特征 {feature_id} ({data_type}) 的semantic_id缓存已存在: {cache_file}")
                print(f"   如需重新计算，请使用 --force_recompute 参数")
                # 显示缓存文件信息
                file_size = cache_file.stat().st_size / (1024 * 1024)  # MB
                print(f"   缓存文件大小: {file_size:.2f} MB")
                continue
            
            print(f"\n🎯 开始处理特征 {feature_id} ({data_type})...")
            print(f"📊 待处理creative_id数量: {len(creative_ids)}")
            
            try:
                # 预计算semantic_id - 🔧 修复：传入正确的多模态数据路径
                semantic_id_dict = precompute_semantic_ids_for_feature(
                    feature_id=feature_id,
                    creative_ids=creative_ids,
                    data_type=data_type,
                    mm_emb_path=mm_emb_path,  # 传入多模态数据路径
                    device=args.device,
                    batch_size=args.batch_size,
                    use_training_strategy=True  # 使用训练时相同的策略
                )
                
                save_semantic_id_dict(semantic_id_dict, feature_id, data_type)
                
                print(f"✅ 特征 {feature_id} ({data_type}) 预计算完成")
                
                # 显示完成统计
                cache_file_size = cache_file.stat().st_size / (1024 * 1024) if cache_file.exists() else 0
                print(f"📊 生成缓存文件大小: {cache_file_size:.2f} MB")
                
            except Exception as e:
                print(f"❌ 特征 {feature_id} ({data_type}) 预计算失败: {e}")
                print(f"🔧 失败的特征ID: {feature_id}")
                print(f"🔧 失败的数据类型: {data_type}")
                print(f"🔧 creative_id数量: {len(creative_ids)}")
                print(f"🔧 多模态数据路径: {mm_emb_path}")
                import traceback
                traceback.print_exc()
                print(f"⚠️  继续处理下一个特征...")
    
    print("\n" + "="*60)
    print("🎉 semantic_id预计算全部完成!")
    print("="*60)
    print("💡 使用方法:")
    print("   训练时使用: --use_precomputed_semantic_ids (会自动加载train缓存)")
    print("   推理时使用: --use_precomputed_semantic_ids (会自动加载eval缓存)")
    print("🚀 预期效果:")
    print("   - 训练和推理速度将大幅提升")
    print("   - 显存占用也会减少")
    print("   - 避免重复的RQ-VAE编码计算")
    
    # 显示缓存文件统计
    cache_dir = os.environ.get('USER_CACHE_PATH')
    if cache_dir:
        cache_path = Path(cache_dir)
        semantic_files = list(cache_path.glob("semantic_id_*.pkl"))
        if semantic_files:
            print(f"\n📊 生成的缓存文件统计:")
            total_size = 0
            for file in semantic_files:
                file_size = file.stat().st_size / (1024 * 1024)  # MB
                total_size += file_size
                print(f"   {file.name}: {file_size:.2f} MB")
            print(f"   总大小: {total_size:.2f} MB")


if __name__ == "__main__":
    main()
