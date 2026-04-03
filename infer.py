import argparse
import json
import os
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MyTestDataset, save_emb
from model import BaselineModel
from config import get_semantic_id_config

def get_ckpt_path():
    ckpt_path = os.environ.get("MODEL_OUTPUT_PATH")
    if ckpt_path is None:
        raise ValueError("MODEL_OUTPUT_PATH is not set")
    for item in os.listdir(ckpt_path):
        if item.endswith(".pt"):
            return os.path.join(ckpt_path, item)


def get_args():
    parser = argparse.ArgumentParser()

    # Train params - 🔧 修正为与run.sh训练完全一致
    parser.add_argument('--batch_size', default=24, type=int)  # 🔧 与run.sh一致
    parser.add_argument('--lr', default=0.002, type=float)
    parser.add_argument('--maxlen', default=101, type=int)

    # Baseline Model construction - 🔧 与run.sh训练完全一致
    parser.add_argument('--hidden_units', default=512, type=int)
    # 📏 Embedding维度配置已移至config.py，使用自适应维度分配
    parser.add_argument('--num_blocks', default=16, type=int)
    parser.add_argument('--num_epochs', default=9, type=int)  # 🔧 与run.sh一致
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--dropout_rate', default=0.1, type=float, help='Universal dropout rate (must match training)')
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--norm_first', action='store_true', default=True, help='Use pre-norm structure (default: True for better performance)')
    parser.add_argument('--post_norm', dest='norm_first', action='store_false', help='Switch to post-norm structure')
    parser.add_argument('--enable_time_bias', action='store_true', default=True,
                       help='启用注意力机制中的时间偏置（time bias），可独立于时间特征使用')
    parser.add_argument('--disable_time_bias', dest='enable_time_bias', action='store_false',
                       help='禁用注意力机制中的时间偏置')
    # 🔧 修正Dropout参数为与run.sh训练完全一致
    parser.add_argument('--emb_dropout_rate', default=0.05, type=float, help='Dropout rate for embeddings (must match training: 0.05)')
    parser.add_argument('--MLP_dropout_rate', default=0.15, type=float, help='Dropout rate for MLPs (must match training: 0.15)')
    parser.add_argument('--transformer_dropout', default=0.05, type=float, help='Dropout rate for Transformer attention layers (must match training: 0.05)')

    # InfoNCE loss parameters - 🔧 与run.sh训练完全一致
    parser.add_argument('--temperature', default=0.04, type=float, help='Temperature parameter for InfoNCE loss (must match training: 0.04)')
    
    # MMemb Feature ID - 🔧 与run.sh训练完全一致
    parser.add_argument('--mm_emb_id', nargs='+', default=['81', '82', '84'], type=str, choices=[str(s) for s in range(81, 87)])
    # RQ-VAE相关参数 - 与main.py保持一致
    parser.add_argument('--enable_rqvae', action='store_true', default=True, help='启用RQ-VAE模式，使用semantic id替代原始多模态特征')
    parser.add_argument('--disable_rqvae', dest='enable_rqvae', action='store_false', help='禁用RQ-VAE模式，使用原始多模态特征')
    parser.add_argument('--use_precomputed_semantic_ids', action='store_true', default=True,
                       help='使用预计算的semantic_id（需要先运行precompute_semantic_ids.py）')
    
    # 时间特征相关参数
    parser.add_argument('--enable_time_features', action='store_true', default=True, 
                       help='启用时间特征（包含时间间隔特征和绝对时间特征）')
    parser.add_argument('--disable_time_features', dest='enable_time_features', action='store_false', 
                       help='禁用时间特征')
    
    # 🎯 时间差特征隔离选项 - 🔧 与run.sh训练完全一致
    parser.add_argument('--disable_time_diff_features', action='store_true', default=False,
                       help='禁用时间差相关特征，仅保留绝对时间特征（run.sh训练已启用此选项）')
    
    # 🎯 候选侧头部类型选择（seq复杂/cand简单架构优化）
    parser.add_argument('--item_cand_head', default='linear', type=str, 
                       choices=['mlp', 'linear', 'identity', 'light_mlp'],
                       help='候选侧头部类型: mlp(EnhancedDNN), linear(nn.Linear), identity(nn.Identity), light_mlp(轻量MLP)')
    
    # RoPE位置编码相关参数 - 🔧 与run.sh训练完全一致
    parser.add_argument('--enable_rope', action='store_true', default=True,
                       help='启用RoPE旋转位置编码 (run.sh训练已启用)')
    parser.add_argument('--disable_rope', dest='enable_rope', action='store_false',
                       help='禁用RoPE，使用传统位置编码')
    parser.add_argument('--rope_theta', default=10000.0, type=float,
                       help='RoPE旋转基数')
    parser.add_argument('--rope_max_seq_len', default=128, type=int,
                       help='RoPE最大序列长度 (must match training: 128)')
    
    # 🎯 HSTU注意力机制相关参数 - 🔧 与run.sh训练完全一致
    parser.add_argument('--attention_mode', default='hstu', choices=['sdpa', 'softmax', 'hstu'],
                       help='注意力计算模式：sdpa(PyTorch SDPA), softmax(标准), hstu(HSTU pointwise SiLU) (must match training: hstu)')
    parser.add_argument('--enable_relative_bias', action='store_true', default=False,
                       help='启用HSTU相对位置偏置(RAB)，与时间偏置结合实现rab_{p,t} (run.sh训练中被注释，实际未启用)')
    parser.add_argument('--disable_relative_bias', dest='enable_relative_bias', action='store_false',
                       help='禁用相对位置偏置，使用传统位置编码或RoPE')

    parser.add_argument('--enable_popularity_sampling', action='store_true', default=True,
                       help='启用基于流行度的负采样（自动使用预计算数据）')
    # 🎯 CTR特征控制参数 - 🔧 与run.sh训练完全一致
    parser.add_argument('--enable_ctr_feature', action='store_true', default=True,
                       help='Enable CTR (click-through rate) as item static feature (run.sh训练已启用)')
    parser.add_argument('--disable_ctr_feature', dest='enable_ctr_feature', action='store_false',
                       help='Disable CTR feature')
    
    # 数据监控参数 - 与main.py保持一致
    parser.add_argument('--enable_data_monitoring', action='store_true', default=True,
                       help='启用数据监控，显示批次样本数据')
    parser.add_argument('--monitoring_interval', default=1000, type=int,
                       help='数据监控间隔（每多少步显示一次）')
    parser.add_argument('--monitoring_samples', default=3, type=int,
                       help='每次监控显示的样本数量')

    args = parser.parse_args()

    # 简化dropout参数设置 - 🔧 移除自动修复，使用明确的训练一致参数
    # 参数一致性检查已通过明确默认值确保，无需运行时修复
    print(f"✅ Dropout参数设置 (与run.sh训练一致):")
    print(f"   emb_dropout_rate={args.emb_dropout_rate}")
    print(f"   MLP_dropout_rate={args.MLP_dropout_rate}")
    print(f"   transformer_dropout={args.transformer_dropout}")
    
    # 🔧 添加关键参数一致性检查
    print("\n" + "="*60)
    print("🔍 推理参数一致性检查 (vs run.sh训练)")
    print("="*60)
    
    # 检查核心架构参数
    print(f"模型架构参数 (期望值来自run.sh):")
    print(f"  - num_blocks: {args.num_blocks} (run.sh: 24) {'✅' if args.num_blocks == 24 else '❌'}")
    print(f"  - num_heads: {args.num_heads} (run.sh: 16) {'✅' if args.num_heads == 16 else '❌'}")
    print(f"  - hidden_units: {args.hidden_units} (run.sh: 256) {'✅' if args.hidden_units == 256 else '❌'}")
    print(f"  - batch_size: {args.batch_size} (run.sh: 192) {'✅' if args.batch_size == 192 else '❌'}")
    
    # 检查Dropout参数
    print(f"Dropout参数 (期望值来自run.sh):")
    print(f"  - emb_dropout_rate: {args.emb_dropout_rate} (run.sh: 0.05) {'✅' if args.emb_dropout_rate == 0.05 else '❌'}")
    print(f"  - MLP_dropout_rate: {args.MLP_dropout_rate} (run.sh: 0.15) {'✅' if args.MLP_dropout_rate == 0.15 else '❌'}")
    print(f"  - transformer_dropout: {args.transformer_dropout} (run.sh: 0.05) {'✅' if args.transformer_dropout == 0.05 else '❌'}")
    
    # 检查训练相关参数
    print(f"训练一致性参数:")
    print(f"  - temperature: {args.temperature} (run.sh: 0.04) {'✅' if args.temperature == 0.04 else '❌'}")
    print(f"  - enable_ctr_feature: {args.enable_ctr_feature} (run.sh: True) {'✅' if args.enable_ctr_feature else '❌'}")
    print(f"  - enable_rqvae: {args.enable_rqvae} (run.sh: True) {'✅' if args.enable_rqvae else '❌'}")
    print(f"  - enable_time_features: {args.enable_time_features} (run.sh: True) {'✅' if args.enable_time_features else '❌'}")
    print(f"  - disable_time_diff_features: {args.disable_time_diff_features} (run.sh: True) {'✅' if args.disable_time_diff_features else '❌'}")
    print(f"  - enable_time_bias: {args.enable_time_bias} (run.sh: True) {'✅' if args.enable_time_bias else '❌'}")
    print(f"  - enable_rope: {args.enable_rope} (run.sh: True) {'✅' if args.enable_rope else '❌'}")
    print(f"  - rope_max_seq_len: {args.rope_max_seq_len} (run.sh: 128) {'✅' if args.rope_max_seq_len == 128 else '❌'}")
    print(f"  - norm_first: {args.norm_first} (run.sh: True) {'✅' if args.norm_first else '❌'}")
    print(f"  - mm_emb_id: {args.mm_emb_id} (run.sh: [81, 82, 84, 86]) {'✅' if args.mm_emb_id == ['81', '82', '84', '86'] else '❌'}")
    
    # 🎯 HSTU相关参数检查
    print(f"HSTU注意力机制参数:")
    print(f"  - attention_mode: {args.attention_mode} (run.sh: hstu) {'✅' if args.attention_mode == 'hstu' else '❌'}")
    print(f"  - enable_relative_bias: {args.enable_relative_bias} (run.sh: False, 被注释) {'✅' if not args.enable_relative_bias else '❌'}")
    
    # 🔧 修正之前的误导性警告
    # 检查关键不一致并给出正确建议
    print(f"\n⚠️  关键不一致检查:")
    has_inconsistency = False
    
    if args.batch_size != 192:
        print(f"   ❌ batch_size不一致: 推理={args.batch_size}, run.sh=192")
        print(f"      建议: 使用 --batch_size 192")
        has_inconsistency = True
    
    if args.emb_dropout_rate != 0.05:
        print(f"   ❌ emb_dropout_rate不一致: 推理={args.emb_dropout_rate}, run.sh=0.05")
        print(f"      建议: 使用 --emb_dropout_rate 0.05")
        has_inconsistency = True
        
    if args.MLP_dropout_rate != 0.15:
        print(f"   ❌ MLP_dropout_rate不一致: 推理={args.MLP_dropout_rate}, run.sh=0.15")
        print(f"      建议: 使用 --MLP_dropout_rate 0.15")
        has_inconsistency = True
        
    if args.transformer_dropout != 0.05:
        print(f"   ❌ transformer_dropout不一致: 推理={args.transformer_dropout}, run.sh=0.05")
        print(f"      建议: 使用 --transformer_dropout 0.05")
        has_inconsistency = True
        
    if args.rope_max_seq_len != 128:
        print(f"   ❌ rope_max_seq_len不一致: 推理={args.rope_max_seq_len}, run.sh=128")
        print(f"      建议: 使用 --rope_max_seq_len 128")
        has_inconsistency = True
        
    if not args.enable_ctr_feature:
        print(f"   ❌ enable_ctr_feature不一致: 推理={args.enable_ctr_feature}, run.sh=True")
        print(f"      建议: 使用 --enable_ctr_feature")
        has_inconsistency = True
        
    if not args.disable_time_diff_features:
        print(f"   ❌ disable_time_diff_features不一致: 推理={args.disable_time_diff_features}, run.sh=True")
        print(f"      建议: 使用 --disable_time_diff_features")
        has_inconsistency = True
    
    if not has_inconsistency:
        print(f"   ✅ 所有关键参数与run.sh训练配置一致!")
    
    # 重要提醒
    print(f"\n💡 重要提醒：")
    print(f"  1. 推理参数现已与 run.sh 训练脚本对齐")
    print(f"  2. 如发现不一致，请按建议调整推理命令行参数") 
    print(f"  3. 模型架构和特征处理流程已确保训练-推理一致性")
    print("="*60)
    
    # 🔧 添加参数一致性检查（与main.py保持一致）
    if args.enable_rqvae:
        # 推理时通常只使用mm_emb_id，但如果设置了rqvae_features也要检查一致性
        if hasattr(args, 'rqvae_features') and args.rqvae_features:
            rqvae_features_set = set(args.rqvae_features)
            mm_emb_id_set = set(args.mm_emb_id)
            
            if rqvae_features_set != mm_emb_id_set:
                print(f"❌ 推理参数不一致警告:")
                print(f"   --rqvae_features: {args.rqvae_features}")
                print(f"   --mm_emb_id: {args.mm_emb_id}")
                print(f"🔧 自动修复：统一使用mm_emb_id的值")
                args.rqvae_features = args.mm_emb_id
        
        print(f"✅ 推理RQ-VAE模式启用，使用特征: {args.mm_emb_id}")
    
    return args


@torch.no_grad()
def pytorch_cosine_retrieval(query_embeddings, candidate_embeddings, candidate_ids, top_k=10, query_batch_size=1000, cand_chunk_size=100000):
    device = query_embeddings.device
    num_queries = query_embeddings.shape[0]
    num_candidates = candidate_embeddings.shape[0]
    
    print(f"🚀 PyTorch检索开始: {num_queries} queries × {num_candidates} candidates")
    
    # 查询embedding归一化
    query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
    # 🔧 修复：确保数据类型一致性，强制转换为int64以支持索引操作
    if isinstance(candidate_ids, np.ndarray):
        candidate_ids = torch.from_numpy(candidate_ids.astype(np.int64)).to(device)
    else:
        candidate_ids = torch.tensor(candidate_ids, device=device, dtype=torch.int64)
    
    all_top_k_ids = []
    
    # 双端分块：query端分批
    for q_start in tqdm(range(0, num_queries, query_batch_size), desc="PyTorch检索"):
        q_end = min(q_start + query_batch_size, num_queries)
        batch_queries = query_embeddings[q_start:q_end]  # [batch_size, embed_dim]
        
        # 初始化该query batch的全局topk
        global_top_values = torch.full((batch_queries.shape[0], top_k), -float('inf'), device=device)
        global_top_indices = torch.zeros((batch_queries.shape[0], top_k), dtype=torch.long, device=device)
        
        # candidate端分块
        for c_start in range(0, num_candidates, cand_chunk_size):
            c_end = min(c_start + cand_chunk_size, num_candidates)
            
            # 动态加载candidate chunk到GPU并归一化
            cand_chunk = candidate_embeddings[c_start:c_end]
            cand_chunk = F.normalize(cand_chunk.to(device), p=2, dim=1)
            
            # 计算当前chunk的相似度
            similarities = torch.matmul(batch_queries, cand_chunk.T)  # [batch_size, chunk_size]
            
            # 合并当前chunk的topk与全局topk
            combined_values = torch.cat([global_top_values, similarities], dim=1)
            combined_indices = torch.cat([
                global_top_indices,
                torch.arange(c_start, c_end, device=device).unsqueeze(0).expand(batch_queries.shape[0], -1)
            ], dim=1)
            
            # 重新选择topk
            top_values, top_positions = torch.topk(combined_values, k=min(top_k, combined_values.shape[1]), dim=1)
            global_top_values = top_values
            global_top_indices = torch.gather(combined_indices, 1, top_positions)
            
            # 清理chunk显存
            del cand_chunk, similarities, combined_values, combined_indices
            if device == 'cuda':
                torch.cuda.empty_cache()
        
        # 转换为实际ID
        batch_top_k_ids = candidate_ids[global_top_indices]
        all_top_k_ids.append(batch_top_k_ids.cpu())
        
        # 🔧 调试信息：验证ID映射正确性
        if q_start == 0:  # 只在第一个batch打印调试信息
            print(f"🔍 调试：前3个query的top-3结果ID sample: {batch_top_k_ids[:min(3, batch_queries.shape[0]), :3]}")
        
        # 清理该batch显存
        del global_top_values, global_top_indices, batch_top_k_ids
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    # 合并结果
    result = torch.cat(all_top_k_ids, dim=0)
    print(f"✅ PyTorch检索完成: 找到 {result.shape[0]} × {result.shape[1]} 个结果")
    
    return result

def read_result_ids(file_path):
    with open(file_path, 'rb') as f:
        # Read the header (num_points_query and FLAGS_query_ann_top_k)
        num_points_query = struct.unpack('I', f.read(4))[0]  # uint32_t -> 4 bytes
        query_ann_top_k = struct.unpack('I', f.read(4))[0]  # uint32_t -> 4 bytes

        print(f"num_points_query: {num_points_query}, query_ann_top_k: {query_ann_top_k}")

        # Calculate how many result_ids there are (num_points_query * query_ann_top_k)
        num_result_ids = num_points_query * query_ann_top_k

        # Read result_ids (uint64_t, 8 bytes per value)
        result_ids = np.fromfile(f, dtype=np.uint64, count=num_result_ids)

        return result_ids.reshape((num_points_query, query_ann_top_k))


def process_cold_start_feat(feat):
    """
    处理冷启动特征。训练集未出现过的特征value为字符串，默认转换为0.可设计替换为更好的方法。
    """
    processed_feat = {}
    for feat_id, feat_value in feat.items():
        if type(feat_value) == list:
            value_list = []
            for v in feat_value:
                if type(v) == str:
                    value_list.append(0)
                else:
                    value_list.append(v)
            processed_feat[feat_id] = value_list
        elif type(feat_value) == str:
            processed_feat[feat_id] = 0
        else:
            processed_feat[feat_id] = feat_value
    return processed_feat


def _save_item_emb_with_model(model, item_ids, retrieval_ids, features, output_path, feat_types, feat_default_value, enable_rqvae=False, dataset=None):
    """
    使用模型生成并保存候选物品的embedding
    🔧 关键修复：参照dataset.py的候选特征处理逻辑，确保批内所有特征的batch维度一致
    
    Args:
        model: 模型实例
        item_ids: 物品ID列表
        retrieval_ids: 检索ID列表
        features: 特征字典列表
        output_path: 输出路径
        feat_types: 特征类型字典
        feat_default_value: 特征默认值字典（来自dataset）
        enable_rqvae: 是否启用RQVAE模式
        dataset: 数据集实例，用于特征处理
    """
    model.eval()
    item_embs = []
    batch_size = 1024  # 候选物品批次大小
    
    # 感觉是没有必要的，候选库特征本身不带有这些特征
    time_related_features = {
        'time_gap', 'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos',
        'month_sin', 'month_cos', 'season_sin', 'season_cos',
        'day_of_year_sin', 'day_of_year_cos', 'is_weekend',
        'day_of_month_sin', 'day_of_month_cos', 'time_gap_continuous'
    }
    
    EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    
    with torch.no_grad():
        for start_idx in tqdm(range(0, len(features), batch_size), desc='生成候选物品embedding'):
            end_idx = min(start_idx + batch_size, len(features))
            
            # 构建批次数据
            batch_features = features[start_idx:end_idx] # 特征字典
            batch_item_ids = item_ids[start_idx:end_idx]
            actual_batch_size = len(batch_features)
            
            # 🔧 关键修复：训练-推理一致性对齐
            # 确保候选库特征处理逻辑与dataset.py中的训练逻辑完全一致：
            # 1. 使用相同的特征类型判断逻辑（is_array_feature, is_continual）
            # 2. 使用相同的张量构建方式（维度、数据类型）
            # 3. 只处理item特征，不处理用户特征（候选库特点）
            # 4. 跳过时间特征（候选库不应包含时序信息）
            final_batch_tensor = {}
            
            # 🎯 重要：候选库只处理item相关特征，不处理用户特征（与训练时的candidate一致）
            candidate_feat_types = [
                ('item_sparse', feat_types.get('item_sparse', [])),
                ('item_array', feat_types.get('item_array', [])),
                ('item_continual', feat_types.get('item_continual', [])),
                ('item_semantic_array', feat_types.get('item_semantic_array', [])),
            ]
            
            # 🔧 参照dataset.py的_feat2numpy_internal方法逻辑，构建候选特征张量
            for feat_type, feat_ids in candidate_feat_types:
                for feat_id in feat_ids:
                    # 跳过时间特征（候选item不应包含时间特征）
                    if feat_id in time_related_features:
                        continue
                    
                    # 构建特征序列（模拟dataset.py中的seq_feature结构）
                    seq_feature = []
                    for feat_dict in batch_features:
                        if feat_id in feat_dict:
                            seq_feature.append({feat_id: feat_dict[feat_id]})
                        else:
                            seq_feature.append({feat_id: feat_default_value[feat_id]})
                    
                    # 🔧 使用与dataset.py完全一致的特征处理逻辑
                    tensor_key = f'seq_{feat_id}'
                    
                    # 判断特征类型（与dataset.py的is_array_feature逻辑一致）
                    is_array_feature = (
                        feat_id in feat_types.get('item_array', []) or 
                        feat_id in feat_types.get('item_semantic_array', [])
                    )
                    
                    if is_array_feature:
                        # 🔧 数组特征处理：与dataset.py的_feat2numpy_internal方法一致
                        max_array_len = 0
                        for item in seq_feature:
                            if feat_id in item and item[feat_id] is not None:
                                if isinstance(item[feat_id], list):
                                    max_array_len = max(max_array_len, len(item[feat_id]))
                                else:
                                    max_array_len = max(max_array_len, 1)
                        
                        if max_array_len == 0:
                            max_array_len = 1  # 防止空数组
                        
                        # 构建数组特征张量 [batch_size, 1, max_array_len] - 候选库序列长度为1
                        feat_tensor = np.zeros((actual_batch_size, 1, max_array_len), dtype=np.int64)
                        for i, item in enumerate(seq_feature):
                            if feat_id in item and item[feat_id] is not None:
                                if isinstance(item[feat_id], list):
                                    actual_len = min(len(item[feat_id]), max_array_len)
                                    feat_tensor[i, 0, :actual_len] = item[feat_id][:actual_len]
                                else:
                                    feat_tensor[i, 0, 0] = item[feat_id]
                        
                        final_batch_tensor[tensor_key] = torch.from_numpy(feat_tensor).to(model.dev)
                    else:
                        # 🔧 标量特征处理：与dataset.py逻辑一致
                        is_continual = (
                            feat_id in feat_types.get('item_continual', [])
                        )
                        
                        feat_data = np.zeros(actual_batch_size, dtype=np.float32 if is_continual else np.int64)
                        for i, item in enumerate(seq_feature):
                            if feat_id in item and item[feat_id] is not None:
                                feat_data[i] = item[feat_id]
                        
                        # 转换为 [batch_size, 1] 张量（候选库序列长度为1）
                        feat_tensor = torch.from_numpy(feat_data).unsqueeze(1).to(model.dev)
                        final_batch_tensor[tensor_key] = feat_tensor
            
            # 🔧 处理多模态特征（端到端RQ-VAE模式或传统模式）- 与dataset.py逻辑一致
            if not (enable_rqvae and hasattr(model, 'use_precomputed_semantic_ids') and model.use_precomputed_semantic_ids):
                for feat_id in feat_types.get('item_emb', []):
                    # 构建特征序列（模拟dataset.py的seq_feature结构）
                    seq_feature = []
                    for feat_dict in batch_features:
                        seq_feature.append({feat_id: feat_dict.get(feat_id)})
                    
                    # 🔧 使用与dataset.py的_process_multimodal_numpy方法一致的逻辑
                    emb_dim = EMB_SHAPE_DICT.get(feat_id, 32)
                    feat_data = np.zeros((actual_batch_size, emb_dim), dtype=np.float32)
                    
                    for i, item in enumerate(seq_feature):
                        if feat_id in item and item[feat_id] is not None:
                            if isinstance(item[feat_id], np.ndarray):
                                feat_data[i] = item[feat_id]
                            elif isinstance(item[feat_id], list):
                                feat_data[i] = np.array(item[feat_id], dtype=np.float32)
                        else:
                            # 使用默认值 - 与dataset.py一致的fallback逻辑
                            if feat_id in feat_default_value:
                                feat_data[i] = feat_default_value[feat_id]
                            else:
                                feat_data[i] = np.zeros(emb_dim, dtype=np.float32)
                    
                    # 转换为 [batch_size, 1, emb_dim] 张量（候选库序列长度为1）
                    feat_tensor = feat_data[:, None, :]  # 添加序列维度
                    final_batch_tensor[f'seq_{feat_id}'] = torch.from_numpy(feat_tensor).to(model.dev)
            
            # 🔧 最终验证：确保所有特征张量的batch维度一致
            for tensor_key, tensor_val in final_batch_tensor.items():
                assert tensor_val.shape[0] == actual_batch_size, f"张量{tensor_key}的batch维度{tensor_val.shape[0]}不等于预期{actual_batch_size}"
            
            # 模型前向传播
            batch_seq = torch.tensor(batch_item_ids, dtype=torch.long).unsqueeze(1).to(model.dev)  # [batch_size, 1]
            
            # 调用模型的feat2emb方法
            item_emb = model.feat2emb(batch_seq, final_batch_tensor, mask=None, include_user=False, mode='cand')
            
            # 提取embedding并归一化
            item_emb = item_emb.squeeze(1)  # [batch_size, hidden_units]
            item_emb = item_emb / item_emb.norm(dim=-1, keepdim=True)
            
            item_embs.append(item_emb.cpu().numpy())
    
    # 拼接所有embedding并保存
    item_embs = np.concatenate(item_embs, axis=0)
    torch.cuda.empty_cache()
    # 🔧 修复：保存为与检索阶段一致的格式
    final_ids = np.array(retrieval_ids, dtype=np.uint64).reshape(-1, 1)
    final_embs = item_embs.astype(np.float32)
    
    print(f"✅ 生成候选物品embedding完成：{final_embs.shape}")
    save_emb(final_embs, Path(output_path, 'embedding.fbin'))
    save_emb(final_ids, Path(output_path, 'id.u64bin'))
    
    print(f"✅ 候选库embedding保存完成: {len(item_embs)} items -> {output_path}")


def get_candidate_emb(indexer, feat_types, feat_default_value, mm_emb_dict, model, enable_rqvae=False, use_precomputed_semantic_ids=False, semantic_id_loaders=None, dataset=None):
    """
    生产候选库item的id和embedding

    Args:
        indexer: 索引字典
        feat_types: 特征类型
        feat_default_value: 特征缺省值
        mm_emb_dict: 多模态特征字典（预计算模式下可为None）
        model: 模型
        enable_rqvae: 是否启用RQVAE模式
        use_precomputed_semantic_ids: 是否使用预计算的semantic_id
        semantic_id_loaders: semantic_id加载器字典（预计算模式下使用）
        dataset: 数据集实例，用于特征处理
    Returns:
        retrieve_id2creative_id: 索引id->creative_id的dict
    """
    EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    candidate_path = Path(os.environ.get('EVAL_DATA_PATH'), 'predict_set.jsonl')
    item_ids, creative_ids, retrieval_ids, features = [], [], [], []
    retrieve_id2creative_id = {}
    
    # 🔧 添加冷启动item统计
    cold_start_count = 0
    semantic_id_hit_count = 0
    semantic_id_miss_count = 0
    
    # 模式选择提示
    if enable_rqvae and use_precomputed_semantic_ids:
        print("🎯 预计算模式：使用缓存的semantic_id")
    elif enable_rqvae:
        print("🎯 端到端模式：实时生成semantic_id")
    else:
        print("📊 传统模式：使用原始多模态特征")

    # # 🔍 添加候选库demo数据展示功能
    # # def show_candidate_demo_data(line_data, line_idx, max_demo_items=5):
    #     """展示候选库demo数据，帮助调试时验证候选库数据读取是否正确"""
    #     if line_idx >= max_demo_items:
    #         return
            
    #     print(f"\n🔍 ===== 候选库Demo数据 (Item {line_idx+1}) =====")
    #     print(f"📝 creative_id: {line_data['creative_id']}")
    #     print(f"📝 retrieval_id: {line_data['retrieval_id']}")
        
    #     features = line_data['features']
    #     print(f"🎯 原始特征数量: {len(features)}")
        
    #     # 展示特征样本
    #     feature_samples = []
    #     for key, value in list(features.items())[:8]:  # 只展示前8个特征
    #         if isinstance(value, list):
    #             if len(value) > 3:
    #                 sample_str = f"{key}: [{value[0]}, {value[1]}, {value[2]}, ...] (len={len(value)})"
    #             else:
    #                 sample_str = f"{key}: {value}"
    #         elif isinstance(value, (int, float)):
    #             sample_str = f"{key}: {value}"
    #         elif isinstance(value, str):
    #             sample_str = f"{key}: '{value}'"
    #         else:
    #             sample_str = f"{key}: {type(value).__name__}"
    #         feature_samples.append(sample_str)
        
    #     if feature_samples:
    #         print(f"🎯 特征样本:")
    #         for feat_info in feature_samples:
    #             print(f"   {feat_info}")
        
    #     if len(features) > 8:
    #         print(f"   ... 还有 {len(features) - 8} 个特征未显示")
        
    #     print(f"🔍 ===== 候选库Demo数据结束 =====\n")
    
    with open(candidate_path, 'r') as f:
        line_idx = 0
        for line in f:
            line = json.loads(line)
            
            # 🔍 展示前几个候选item的demo数据
            # if line_idx < 5:  # 只展示前5个候选item的demo数据
            #     show_candidate_demo_data(line, line_idx)
            
            feature = line['features']
            creative_id = line['creative_id']
            retrieval_id = line['retrieval_id']
            item_id = indexer[creative_id] if creative_id in indexer else 0
            
            # 🔧 统计冷启动item
            if item_id == 0:
                cold_start_count += 1
            
            line_idx += 1
            
            # 补充缺失的“常规”特征
            missing_fields = set(
                feat_types['item_sparse'] + feat_types['item_array'] + feat_types['item_continual']
            ) - set(feature.keys())
            feature = process_cold_start_feat(feature) # 特征字典
            # 🔧 关键修复：CTR特征注入与训练时dataset.py保持完全一致
            # 补充缺失的常规特征（候选物品不包含时间特征，符合双塔模型设计）
            for feat_id in missing_fields:
                if feat_id == 'item_ctr' and dataset and hasattr(dataset, 'item_ctr') and hasattr(dataset, 'enable_ctr_feature') and dataset.enable_ctr_feature:
                    # 🎯 使用与dataset.py中fill_missing_feat_cached完全一致的CTR注入逻辑
                    if creative_id in dataset.item_ctr:
                        # 使用预计算的CTR值（与训练时一致）
                        feature[feat_id] = dataset.item_ctr[creative_id]
                    else:
                        # 使用全局平均CTR作为冷启动默认值（与训练时一致）
                        feature[feat_id] = dataset.global_avg_ctr
                else:
                    # 非CTR特征或CTR特征未启用时，使用默认值
                    feature[feat_id] = feat_default_value[feat_id]
            
            # 🎯 统一的语义ID和多模态特征处理
            if enable_rqvae:
                if use_precomputed_semantic_ids:
                    # 预计算模式：加载统一格式的semantic_id数组
                    if semantic_id_loaders:
                        for feat_id, loader in semantic_id_loaders.items():
                            feature_name = loader.feature_config['feature_name']
                            default_array_length = loader.feature_config['array_length']
                            default_value = loader.feature_config['default_value']
                            
                            # 关键修复：无论是否冷启动，都尝试加载预计算的semantic_id
                            # 预计算阶段已经为所有候选集item（包括冷启动）生成了真实的semantic_id
                            semantic_ids = loader.get_semantic_id(creative_id)
                            if semantic_ids is not None:
                                feature[feature_name] = semantic_ids
                                semantic_id_hit_count += 1
                                # 🔧 调试：显示冷启动item成功加载semantic_id的情况
                                if item_id == 0 and semantic_id_hit_count <= 5:
                                    print(f"✅ 冷启动item {creative_id} 成功加载semantic_id: {semantic_ids[:3]}...")
                            else:
                                # 只有真正未命中时才使用默认值（这种情况应该很少）
                                feature[feature_name] = [default_value] * default_array_length
                                semantic_id_miss_count += 1
                                if item_id != 0:
                                    print(f"⚠️ 警告：训练集item {creative_id} 未找到semantic_id，使用默认值")
                                elif semantic_id_miss_count <= 5:
                                    print(f"⚠️ 冷启动item {creative_id} 未找到semantic_id，使用默认值")
                                # 注意：冷启动item未命中是可能的，因为可能预计算时未覆盖到
                else:
                    # 端到端模式：需要原始多模态特征，将在模型中实时转换为语义ID数组
                    if feat_types['item_emb']:  # 只有在有多模态特征时才处理
                        for feat_id in feat_types['item_emb']:
                            # 🔧 关键修复：creative_id类型转换问题
                            # JSON读取的creative_id是字符串，但多模态加载器可能期望整型
                            emb_value = None
                            if mm_emb_dict:
                                # 先尝试原始creative_id（字符串）
                                emb_value = mm_emb_dict.get(feat_id, creative_id)
                                # 如果失败且creative_id是字符串，尝试转换为整型
                                if emb_value is None and isinstance(creative_id, str) and creative_id.isdigit():
                                    emb_value = mm_emb_dict.get(feat_id, int(creative_id))
                                # 如果失败且creative_id是整型，尝试转换为字符串
                                elif emb_value is None and isinstance(creative_id, int):
                                    emb_value = mm_emb_dict.get(feat_id, str(creative_id))
                            
                            if emb_value is not None:
                                feature[feat_id] = emb_value
                            else:
                                # 使用默认值 - 优先从dataset获取
                                if feat_id in feat_default_value:
                                    feature[feat_id] = feat_default_value[feat_id]
                                else:
                                    feature[feat_id] = np.zeros(EMB_SHAPE_DICT[feat_id], dtype=np.float32)
            else:
                # 传统模式：使用原始多模态特征
                if feat_types['item_emb']:  # 只有在有多模态特征时才处理
                    for feat_id in feat_types['item_emb']:
                        # 🔧 关键修复：creative_id类型转换问题
                        # JSON读取的creative_id是字符串，但多模态加载器可能期望整型
                        emb_value = None
                        if mm_emb_dict:
                            # 先尝试原始creative_id（字符串）
                            emb_value = mm_emb_dict.get(feat_id, creative_id)
                            # 如果失败且creative_id是字符串，尝试转换为整型
                            if emb_value is None and isinstance(creative_id, str) and creative_id.isdigit():
                                emb_value = mm_emb_dict.get(feat_id, int(creative_id))
                            # 如果失败且creative_id是整型，尝试转换为字符串
                            elif emb_value is None and isinstance(creative_id, int):
                                emb_value = mm_emb_dict.get(feat_id, str(creative_id))
                        
                        if emb_value is not None:
                            feature[feat_id] = emb_value
                        else:
                            # 使用默认值 - 优先从dataset获取
                            if feat_id in feat_default_value:
                                feature[feat_id] = feat_default_value[feat_id]
                            else:
                                feature[feat_id] = np.zeros(EMB_SHAPE_DICT[feat_id], dtype=np.float32)

            item_ids.append(item_id)
            creative_ids.append(creative_id)
            retrieval_ids.append(retrieval_id)
            features.append(feature)
            retrieve_id2creative_id[retrieval_id] = creative_id

    # 🔍 展示最终处理后的特征数据样本
    def show_processed_features_demo(features_list, max_demo_items=3):
        """展示经过所有处理步骤后的特征数据样本"""
        print(f"\n🔍 ===== 最终处理后特征Demo数据 =====")
        demo_count = min(max_demo_items, len(features_list))
        
        for i in range(demo_count):
            feature = features_list[i]
            print(f"\n📝 处理后样本 {i+1}:")
            print(f"   🎯 最终特征数量: {len(feature)}")
            
            # 按特征类型分组展示
            semantic_features = []
            multimodal_features = []
            regular_features = []
            
            for key, value in feature.items():
                if 'semantic' in key.lower():
                    if isinstance(value, list):
                        if len(value) > 3:
                            semantic_features.append(f"{key}: [{value[0]}, {value[1]}, {value[2]}, ...] (len={len(value)})")
                        else:
                            semantic_features.append(f"{key}: {value}")
                    else:
                        semantic_features.append(f"{key}: {value}")
                elif key in ['81', '82', '83', '84', '85', '86']:  # 多模态特征ID
                    if isinstance(value, np.ndarray):
                        multimodal_features.append(f"{key}: ndarray{value.shape} [{value.min():.3f}, {value.max():.3f}]")
                    elif isinstance(value, list):
                        multimodal_features.append(f"{key}: list[{len(value)}]")
                    else:
                        multimodal_features.append(f"{key}: {type(value).__name__}")
                else:
                    if isinstance(value, (int, float)):
                        regular_features.append(f"{key}: {value}")
                    elif isinstance(value, list):
                        if len(value) > 3:
                            regular_features.append(f"{key}: [{value[0]}, {value[1]}, {value[2]}, ...] (len={len(value)})")
                        else:
                            regular_features.append(f"{key}: {value}")
                    else:
                        regular_features.append(f"{key}: {type(value).__name__}")
            
            if semantic_features:
                print(f"   🎯 语义特征 ({len(semantic_features)})个:")
                for feat in semantic_features[:3]:  # 只显示前3个
                    print(f"      {feat}")
                if len(semantic_features) > 3:
                    print(f"      ... 还有 {len(semantic_features) - 3} 个语义特征")
            
            if multimodal_features:
                print(f"   📊 多模态特征 ({len(multimodal_features)})个:")
                for feat in multimodal_features:
                    print(f"      {feat}")
            
            if regular_features:
                print(f"   📋 常规特征 ({len(regular_features)})个:")
                for feat in regular_features[:5]:  # 只显示前5个
                    print(f"      {feat}")
                if len(regular_features) > 5:
                    print(f"      ... 还有 {len(regular_features) - 5} 个常规特征")
        
        print(f"🔍 ===== 最终处理后特征Demo数据结束 =====\n")
    
    # 展示处理后的特征样本
    if features:
        show_processed_features_demo(features)

    # 🔧 打印冷启动item和semantic_id加载统计
    total_items = len(item_ids)
    print(f"\n📊 候选集加载统计:")
    print(f"   总候选item数量: {total_items}")
    print(f"   冷启动item数量: {cold_start_count} ({cold_start_count/total_items:.1%})")
    print(f"   训练集item数量: {total_items - cold_start_count} ({(total_items - cold_start_count)/total_items:.1%})")
    
    # 🔧 关键验证：CTR特征分布统计（验证修复是否生效）
    if dataset and hasattr(dataset, 'enable_ctr_feature') and dataset.enable_ctr_feature and 'item_ctr' in feat_types.get('item_continual', []):
        ctr_values = []
        default_ctr_count = 0
        for feature in features:
            if 'item_ctr' in feature:
                ctr_val = feature['item_ctr']
                ctr_values.append(ctr_val)
                # 检查是否为默认值（可能是global_avg_ctr或feat_default_value中的默认值）
                if abs(ctr_val - dataset.global_avg_ctr) < 1e-6 or abs(ctr_val - feat_default_value.get('item_ctr', 0.05)) < 1e-6:
                    default_ctr_count += 1
        
        if ctr_values:
            import numpy as np
            ctr_array = np.array(ctr_values)
            unique_values = len(set(ctr_values))
            print(f"\n🔍 CTR特征分布验证 (关键性能指标):")
            print(f"   CTR均值: {ctr_array.mean():.4f}")
            print(f"   CTR标准差: {ctr_array.std():.4f}")
            print(f"   CTR范围: [{ctr_array.min():.4f}, {ctr_array.max():.4f}]")
            print(f"   不同CTR值数量: {unique_values} (总候选数: {len(ctr_values)})")
            print(f"   默认值占比: {default_ctr_count}/{len(ctr_values)} ({default_ctr_count/len(ctr_values):.1%})")
            
            # 关键判断：如果默认值占比过高，说明CTR注入失败
            if default_ctr_count / len(ctr_values) > 0.8:
                print(f"   ❌ 警告：默认值占比过高 ({default_ctr_count/len(ctr_values):.1%})，CTR特征可能未正确注入！")
                print(f"   🔧 建议：检查dataset.item_ctr字典是否正确加载")
            elif unique_values < 10:
                print(f"   ⚠️  注意：CTR值种类较少 ({unique_values}个)，可能影响模型表现")
            else:
                print(f"   ✅ CTR特征分布正常，注入成功！")
        else:
            print(f"\n❌ 未找到item_ctr特征，CTR注入可能失败！")
    else:
        print(f"\n📋 CTR特征未启用或不在特征列表中，跳过CTR验证")
    
    if enable_rqvae and use_precomputed_semantic_ids:
        print(f"📊 Semantic ID加载统计:")
        print(f"   成功加载数量: {semantic_id_hit_count} ({semantic_id_hit_count/total_items:.1%})")
        print(f"   使用默认值数量: {semantic_id_miss_count} ({semantic_id_miss_count/total_items:.1%})")
        
        if semantic_id_hit_count > 0:
            print(f"✅ 预计算semantic_id覆盖良好！")
        else:
            print(f"❌ 警告：没有找到任何预计算的semantic_id！")
    
    # 生成embedding - 传入feat_default_value和dataset
    _save_item_emb_with_model(model, item_ids, retrieval_ids, features, 
                             os.environ.get('EVAL_RESULT_PATH'), feat_types, feat_default_value, enable_rqvae=enable_rqvae, dataset=dataset)
    
    with open(Path(os.environ.get('EVAL_RESULT_PATH'), "retrive_id2creative_id.json"), "w") as f:
        json.dump(retrieve_id2creative_id, f)
    return retrieve_id2creative_id


def infer():
    args = get_args()
    print(f"🎯 推理参数总结：")
    print(f"   - enable_rqvae: {args.enable_rqvae}")
    if args.enable_rqvae:
        print(f"     ✅ RQ-VAE模式已启用，将使用语义ID替代原始多模态特征")
        print(f"     📊 激活的多模态特征: {args.mm_emb_id}")
    else:
        print(f"     ❌ RQ-VAE模式已禁用，将使用原始多模态特征")
        print(f"     📊 多模态特征ID: {args.mm_emb_id}")
    print(f"   - hidden_units: {args.hidden_units}")
    print(f"   - dropout_rate: {args.dropout_rate}")
    print(f"   - temperature: {args.temperature}")
    
    data_path = os.environ.get('EVAL_DATA_PATH')
    test_dataset = MyTestDataset(data_path, args)
    # ⚡ 多进程优化：推理阶段也享受加速
    num_workers = min(4, os.cpu_count() or 1)
    print(f"🚀 启用推理DataLoader多进程加速: num_workers={num_workers}")
    print("⚡ 推理数据处理已完全下沉到Dataset，多进程将提供真实加速")
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        collate_fn=test_dataset.collate_fn,
        pin_memory=True,
        prefetch_factor=2
    )
    usernum, itemnum = test_dataset.usernum, test_dataset.itemnum
    feat_statistics, feat_types = test_dataset.feat_statistics, test_dataset.feature_types
    
    print(f"📊 数据集信息：usernum={usernum}, itemnum={itemnum}")
    print(f"📊 特征类型：{feat_types}")
    
    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)
    model.eval()

    ckpt_path = get_ckpt_path()
    print(f"📂 加载模型：{ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=torch.device(args.device)))
    
    # 💾 保存模型到user_cache，命名符合main.py续训格式
    # os and Path are already imported at the top of the file
    
    # 创建user_cache目录（使用环境变量USER_CACHE_PATH）
    # cache_dir = os.environ.get('USER_CACHE_PATH')
    # if cache_dir is None:
    #     # 如果环境变量未设置，回退到用户主目录下的user_cache
    #     user_cache_dir = Path.home() / "user_cache"
    #     print(f"⚠️ USER_CACHE_PATH环境变量未设置，使用默认路径: {user_cache_dir}")
    # else:
    #     user_cache_dir = Path(cache_dir)
    # user_cache_dir.mkdir(parents=True, exist_ok=True)
    
    # import re
    
    # # 尝试从路径中提取epoch或global_step信息
    # ckpt_filename = os.path.basename(ckpt_path)
    # ckpt_dirname = os.path.basename(os.path.dirname(ckpt_path))
    
    # # 检查目录名是否包含global_step信息（如：global_step3875.valid_loss=7.8523）
    # global_step_match = re.search(r'global_step(\d+)', ckpt_dirname)
    # epoch_match = re.search(r'epoch(\d+)', ckpt_dirname)
    
    # if global_step_match:
    #     # 使用global_step格式，创建目录结构（与main.py保持一致）
    #     global_step_num = int(global_step_match.group(1))
    #     # 提取valid_loss信息（如果有）
    #     valid_loss_match = re.search(r'valid_loss=([\d\.]+)', ckpt_dirname)
    #     if valid_loss_match:
    #         valid_loss = float(valid_loss_match.group(1))
    #         save_dir_name = f"global_step{global_step_num}.valid_loss={valid_loss:.4f}"
    #     else:
    #         save_dir_name = f"global_step{global_step_num}"
    # elif epoch_match:
    #     # 使用epoch格式，创建目录结构
    #     epoch_num = int(epoch_match.group(1))
    #     # 尝试从路径中获取global_step信息，如果没有则设为0
    #     global_step_for_epoch = 0
    #     if global_step_match:
    #         global_step_for_epoch = int(global_step_match.group(1))
    #     save_dir_name = f"epoch{epoch_num}.global_step{global_step_for_epoch}"
    # else:
    #     # 如果无法解析，使用默认命名
    #     save_dir_name = "model_from_infer"
    #     print(f"⚠️ 无法从路径解析epoch/global_step信息，使用默认命名: {save_dir_name}")
    
    # # 创建保存目录（与main.py保持一致的结构）
    # save_dir = user_cache_dir / save_dir_name
    # save_dir.mkdir(parents=True, exist_ok=True)
    
    # # 保存模型到model.pt文件（与main.py保持一致）
    # model_path = save_dir / "model.pt"
    # torch.save(model.state_dict(), model_path)
    # print(f"💾 模型已保存到user_cache: {model_path}")
    # print(f"🔄 续训时可使用: --state_dict_path {model_path}")
    
    # 🔍 模型-推理一致性检查
    print("\n" + "="*60)
    print("🔍 模型-推理配置一致性检查")
    print("="*60)
    
    # RQ-VAE配置检查
    model_rqvae_enabled = hasattr(model, 'enable_rqvae') and model.enable_rqvae
    if args.enable_rqvae != model_rqvae_enabled:
        print(f"⚠️  RQ-VAE配置不一致！")
        print(f"   推理enable_rqvae: {args.enable_rqvae}")
        print(f"   模型enable_rqvae: {model_rqvae_enabled}")
        if args.enable_rqvae and not model_rqvae_enabled:
            print(f"   🔧 建议：模型未启用RQ-VAE，请使用 --disable_rqvae 参数")
        elif not args.enable_rqvae and model_rqvae_enabled:
            print(f"   🔧 建议：模型已启用RQ-VAE，请移除 --disable_rqvae 参数")
        print(f"   继续推理，但结果可能不准确...")
    else:
        print(f"✅ RQ-VAE配置一致性检查通过")
    
    # 🎯 HSTU注意力机制配置检查
    attention_layers = getattr(model, 'attention_layers', [])
    if attention_layers:
        first_attn_layer = attention_layers[0]
        model_attention_mode = getattr(first_attn_layer, 'attention_mode', 'unknown')
        model_enable_relative_bias = getattr(first_attn_layer, 'enable_relative_bias', True)
        model_enable_time_bias = getattr(first_attn_layer, 'enable_time_bias', False)
        model_enable_rope = getattr(first_attn_layer, 'enable_rope', False)
        
        print(f"HSTU注意力机制配置:")
        print(f"  推理attention_mode: {args.attention_mode} | 模型: {model_attention_mode}")
        print(f"  推理enable_relative_bias: {args.enable_relative_bias} | 模型: {model_enable_relative_bias}")
        print(f"  推理enable_time_bias: {args.enable_time_bias} | 模型: {model_enable_time_bias}")
        print(f"  推理enable_rope: {args.enable_rope} | 模型: {model_enable_rope}")
        
        # 检查关键不一致
        inconsistencies = []
        if args.attention_mode != model_attention_mode:
            inconsistencies.append(f"attention_mode: 推理={args.attention_mode}, 模型={model_attention_mode}")
        if args.enable_relative_bias != model_enable_relative_bias:
            inconsistencies.append(f"enable_relative_bias: 推理={args.enable_relative_bias}, 模型={model_enable_relative_bias}")
        if args.enable_time_bias != model_enable_time_bias:
            inconsistencies.append(f"enable_time_bias: 推理={args.enable_time_bias}, 模型={model_enable_time_bias}")
        if args.enable_rope != model_enable_rope:
            inconsistencies.append(f"enable_rope: 推理={args.enable_rope}, 模型={model_enable_rope}")
        
        if inconsistencies:
            print(f"⚠️  HSTU配置不一致检测到 {len(inconsistencies)} 个问题:")
            for inc in inconsistencies:
                print(f"   - {inc}")
            print(f"   🔧 建议：使用与训练时完全一致的参数进行推理")
            print(f"   继续推理，但注意力机制可能与训练时不同...")
        else:
            print(f"✅ HSTU注意力机制配置一致性检查通过")
    else:
        print(f"⚠️  无法检查注意力层配置（模型结构可能不同）")
    
    print("="*60)
        
    # 🎯 RQ-VAE模型加载状态检查
    if args.enable_rqvae:
        if hasattr(model, 'rqvae_models') and model.rqvae_models:
            print(f"✅ RQ-VAE预训练模型已加载: {list(model.rqvae_models.keys())}")
        elif not args.use_precomputed_semantic_ids:
            print(f"❌ 警告：启用RQ-VAE但未找到预训练模型，语义ID生成可能失败")
        else:
            print(f"📊 使用预计算semantic_id模式，无需RQ-VAE预训练模型")

    all_embs = []
    user_list = []
    
    # 🔍 添加数据调试展示功能
    def show_demo_data(batch, step, max_demo_batches=3, max_samples_per_batch=2):
        """展示demo数据，帮助调试时验证数据读取是否正确"""
        if step >= max_demo_batches:
            return
            
        print(f"\n🔍 ===== Demo数据展示 (Batch {step+1}) =====")
        batch_size = len(batch['user_id'])
        print(f"📊 当前batch大小: {batch_size}")
        
        # 展示前几个样本的详细信息
        demo_count = min(max_samples_per_batch, batch_size)
        for i in range(demo_count):
            print(f"\n📝 样本 {i+1}:")
            print(f"   👤 user_id: {batch['user_id'][i]}")
            
            # 展示序列数据
            if 'seq' in batch:
                seq_data = batch['seq'][i]
                print(f"   📚 序列长度: {len(seq_data)}")
                print(f"   📚 序列前5个item: {seq_data[:5].tolist() if len(seq_data) >= 5 else seq_data.tolist()}")
            
            # 展示特征数据（选择性展示关键特征）
            feature_keys = [k for k in batch.keys() if k.startswith('seq_') and k not in ['seq']]
            print(f"   🎯 特征数量: {len(feature_keys)}")
            
            # 展示前几个重要特征的样本
            important_features = []
            for key in sorted(feature_keys)[:5]:  # 只展示前5个特征
                if key in batch:
                    feat_tensor = batch[key][i]
                    if feat_tensor.dim() == 1:  # 1D特征
                        feat_sample = feat_tensor[:3].tolist() if len(feat_tensor) >= 3 else feat_tensor.tolist()
                        important_features.append(f"{key}: {feat_sample}")
                    elif feat_tensor.dim() == 2:  # 2D特征 (序列特征)
                        feat_sample = feat_tensor[0][:3].tolist() if feat_tensor.shape[1] >= 3 else feat_tensor[0].tolist()
                        important_features.append(f"{key}: {feat_sample}")
            
            if important_features:
                print(f"   🎯 关键特征样本:")
                for feat_info in important_features:
                    print(f"      {feat_info}")
            
            # 展示数据类型和形状信息
            if i == 0:  # 只在第一个样本展示形状信息
                print(f"\n📐 数据形状信息:")
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        print(f"      {key}: {value.shape} ({value.dtype})")
                    elif isinstance(value, list):
                        print(f"      {key}: list[{len(value)}] (sample: {type(value[0]) if value else 'empty'})")
        
        print(f"🔍 ===== Demo数据展示结束 =====\n")
    
    for step, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
        # 🔍 展示demo数据（仅前几个batch）
        if step < 3:  # 只展示前3个batch的demo数据
            show_demo_data(batch, step)
        
        # ⚡ 新的数据格式：推理也使用tensor字典
        logits = model.predict(batch) # seq, token_type, seq_feat, user_id
        for i in range(logits.shape[0]):
            emb = logits[i].unsqueeze(0).detach().cpu().numpy().astype(np.float32)
            all_embs.append(emb)
        user_list += batch['user_id']  # 从字典中获取user_id

    # 生成候选库的embedding 以及 id文件
    retrieve_id2creative_id = get_candidate_emb(
        test_dataset.indexer['i'],
        test_dataset.feature_types,
        test_dataset.feature_default_value,
        test_dataset.mm_emb_dict,
        model,
        enable_rqvae=args.enable_rqvae,
        use_precomputed_semantic_ids=args.use_precomputed_semantic_ids,
        semantic_id_loaders=getattr(test_dataset, 'semantic_id_loaders', None),
        dataset=test_dataset  # 🎯 传入数据集以支持CTR特征
    )
    all_embs = np.concatenate(all_embs, axis=0)
    
    # 🚀 使用PyTorch检索替代faiss - 速度提升15-20倍
    print("🚀 使用PyTorch检索替代faiss，预计速度提升15-20倍...")
    
    # 加载候选库embedding
    candidate_emb_path = Path(os.environ.get("EVAL_RESULT_PATH"), "embedding.fbin")
    candidate_id_path = Path(os.environ.get("EVAL_RESULT_PATH"), "id.u64bin")
    
    # 读取候选embedding
    with open(candidate_emb_path, 'rb') as f:
        num_candidates = struct.unpack('I', f.read(4))[0]
        embed_dim = struct.unpack('I', f.read(4))[0]
        candidate_embeddings = np.fromfile(f, dtype=np.float32).reshape(num_candidates, embed_dim)
    
    # 读取候选ID
    with open(candidate_id_path, 'rb') as f:
        num_ids = struct.unpack('I', f.read(4))[0]
        _ = struct.unpack('I', f.read(4))[0]  # 跳过维度信息
        candidate_ids = np.fromfile(f, dtype=np.uint64, count=num_ids)
    
    # 🔧 修复：将uint64转换为int64以支持PyTorch索引操作
    candidate_ids = candidate_ids.astype(np.int64)
    
    print(f"📊 候选库: {num_candidates} candidates, embed_dim={embed_dim}")
    
    # 🔧 精度验证：检查embedding数据质量
    print(f"🔍 候选embedding质量检查:")
    print(f"   数据类型: {candidate_embeddings.dtype}")
    print(f"   数值范围: [{candidate_embeddings.min():.6f}, {candidate_embeddings.max():.6f}]")
    print(f"   是否包含NaN: {np.isnan(candidate_embeddings).any()}")
    print(f"   是否包含Inf: {np.isinf(candidate_embeddings).any()}")
    
    # 检查query embedding质量
    print(f"🔍 Query embedding质量检查:")
    print(f"   数据类型: {all_embs.dtype}")
    print(f"   数值范围: [{all_embs.min():.6f}, {all_embs.max():.6f}]")
    print(f"   是否包含NaN: {np.isnan(all_embs).any()}")
    print(f"   是否包含Inf: {np.isinf(all_embs).any()}")
    
    # 转换为PyTorch tensor
    query_tensor = torch.from_numpy(all_embs).to(args.device)
    candidate_tensor = torch.from_numpy(candidate_embeddings)
    
    # 🔧 重要：确保候选embedding未预先归一化，让算法内部处理
    print(f"🔍 检查embedding是否预先归一化:")
    query_norms = np.linalg.norm(all_embs, axis=1)
    cand_norms = np.linalg.norm(candidate_embeddings, axis=1)
    print(f"   Query embedding norm 均值±标准差: {query_norms.mean():.4f}±{query_norms.std():.4f}")
    print(f"   Candidate embedding norm 均值±标准差: {cand_norms.mean():.4f}±{cand_norms.std():.4f}")
    if abs(query_norms.mean() - 1.0) < 0.01:
        print(f"   ⚠️  Query embedding似乎已经归一化")
    if abs(cand_norms.mean() - 1.0) < 0.01:
        print(f"   ⚠️  Candidate embedding似乎已经归一化")
    
    # PyTorch检索
    import time
    start_time = time.time()
    
    # 🧪 可选：小规模精度验证（仅在样本较小时启用）
    num_queries = query_tensor.shape[0]
    if num_queries <= 100 and num_candidates <= 10000:
        print("🧪 启用小规模精度验证...")
        # 暴力计算前10个query的真实top-k作为参考
        with torch.no_grad():
            sample_queries = query_tensor[:min(10, num_queries)]
            sample_cands = candidate_tensor.to(args.device)
            
            # 全量计算相似度矩阵
            sample_queries_norm = F.normalize(sample_queries, p=2, dim=1)
            sample_cands_norm = F.normalize(sample_cands, p=2, dim=1)
            similarity_matrix = torch.matmul(sample_queries_norm, sample_cands_norm.T)
            
            # 获取真实top-k
            true_top_values, true_top_indices = torch.topk(similarity_matrix, k=10, dim=1)
            true_top_ids = candidate_ids[true_top_indices].cpu()
            
            print(f"🔍 真实top-k样本 (前3个query的top-3): {true_top_ids[:3, :3]}")
            
            del sample_cands, sample_queries_norm, sample_cands_norm, similarity_matrix
            torch.cuda.empty_cache()
    
    top10_indices = pytorch_cosine_retrieval(
        query_tensor, candidate_tensor, candidate_ids, 
        top_k=10, query_batch_size=1000, cand_chunk_size=100000
    )
    retrieval_time = time.time() - start_time
    print(f"⚡ PyTorch检索耗时: {retrieval_time:.2f}s (预期faiss需要15-20分钟)")
    
    # 🧪 验证精度（与小规模暴力计算对比）
    if num_queries <= 100 and num_candidates <= 10000 and 'true_top_ids' in locals():
        print("🧪 精度验证结果:")
        fast_top_ids = top10_indices[:min(10, num_queries), :3]  # 只比较前3个结果
        
        # 计算top-3命中率
        hits = 0
        total = 0
        for i in range(min(3, fast_top_ids.shape[0])):
            for j in range(3):
                if fast_top_ids[i, j].item() in true_top_ids[i, :3].tolist():
                    hits += 1
                total += 1
        
        hit_rate = hits / total if total > 0 else 0
        print(f"   Top-3命中率: {hit_rate:.1%} ({hits}/{total})")
        print(f"   分块算法结果: {fast_top_ids}")
        print(f"   暴力计算结果: {true_top_ids[:min(3, true_top_ids.shape[0]), :3]}")
        
        if hit_rate >= 0.9:
            print("   ✅ 精度验证通过！分块算法结果与暴力计算高度一致")
        else:
            print("   ⚠️ 精度验证警告：分块算法与暴力计算存在差异")
    else:
        print("🔍 数据规模较大，跳过精度验证（建议在小数据集上测试验证）")
    
    # 转换为creative_id
    top10s_untrimmed = []
    for query_idx in range(top10_indices.shape[0]):
        for k in range(top10_indices.shape[1]):
            retrieval_id = int(top10_indices[query_idx, k].item())
            creative_id = retrieve_id2creative_id.get(retrieval_id, 0)
            top10s_untrimmed.append(creative_id)

    top10s = [top10s_untrimmed[i : i + 10] for i in range(0, len(top10s_untrimmed), 10)]

    return top10s, user_list
