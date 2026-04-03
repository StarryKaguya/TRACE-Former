import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import MyDataset
from model import BaselineModel

def get_dynamic_exposure_weight(args, current_step, total_steps):
    """
    计算动态的exposure_weight，实现前期高后期低的策略
    
    Args:
        args: 命令行参数
        current_step: 当前训练步数
        total_steps: 总训练步数
        
    Returns:
        float: 当前应使用的exposure_weight
    """
    # 如果没有设置动态调整，使用固定值
    if args.exposure_weight_start is None or args.exposure_weight_end is None:
        return args.exposure_weight
    
    # 计算进度比例 [0, 1]
    progress = min(current_step / max(total_steps, 1), 1.0)
    
    start_weight = args.exposure_weight_start
    end_weight = args.exposure_weight_end
    
    if args.exposure_decay_strategy == 'linear':
        # 线性衰减：0.8 -> 0.4
        current_weight = start_weight + (end_weight - start_weight) * progress
    elif args.exposure_decay_strategy == 'cosine':
        # 余弦衰减（平滑过渡）
        current_weight = end_weight + (start_weight - end_weight) * 0.5 * (1 + np.cos(np.pi * progress))
    elif args.exposure_decay_strategy == 'exponential':
        # 指数衰减（前期缓慢，后期快速）
        if start_weight > 0:
            decay_rate = np.log(end_weight / start_weight)
            current_weight = start_weight * np.exp(decay_rate * progress)
        else:
            current_weight = start_weight
    else:
        current_weight = args.exposure_weight
    
    return max(0.0, min(1.0, current_weight))  # 确保在[0,1]范围内


def get_args():
    parser = argparse.ArgumentParser()

    # Train params
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.002, type=float)
    parser.add_argument('--maxlen', default=101, type=int)
    
    # Training stability params
    parser.add_argument('--weight_decay', default=0.01, type=float, help='Weight decay for AdamW optimizer')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='Max gradient norm for clipping, <=0 to disable')
    parser.add_argument('--adaptive_grad_clip', action='store_true', help='Enable adaptive gradient clipping')
    parser.add_argument('--warmup_steps', default=1000, type=int, help='Learning rate warmup steps')
    parser.add_argument('--lr_schedule', default='cosine_with_restarts', type=str, 
                        choices=['constant', 'cosine', 'cosine_with_restarts', 'polynomial'], 
                        help='Learning rate schedule after warmup')
    parser.add_argument('--min_lr_ratio', default=0.1, type=float, help='Minimum lr as ratio of initial lr')
    parser.add_argument('--exposure_weight', default=0.6, type=float, help='Weight for exposure samples (0.0-1.0)')
    
    # 全量训练控制
    parser.add_argument('--full_train', action='store_true', default=False,
                       help='启用全量训练模式，不划分验证集，trust public leaderboard')
    parser.add_argument('--valid_ratio', default=0.01, type=float,
                       help='验证集比例，仅在非全量训练模式下有效 (default: 0.1 即9:1划分)')
    
    # 动态exposure_weight控制 
    parser.add_argument('--exposure_weight_start', default=None, type=float, help='初始曝光权重，启用动态调整')
    parser.add_argument('--exposure_weight_end', default=None, type=float, help='最终曝光权重，启用动态调整')
    parser.add_argument('--exposure_decay_strategy', default='linear', choices=['linear', 'cosine', 'exponential'], 
                       help='曝光权重衰减策略')
    
    parser.add_argument('--enable_popularity_sampling', action='store_true', default=False, 
                       help='Enable popularity-aware negative sampling (automatically uses precomputed data)')
    parser.add_argument('--enable_alias_sampling', action='store_true', default=True, 
                       help='Enable alias method for O(1) weighted sampling (only effective with --enable_popularity_sampling)')
    parser.add_argument('--disable_alias_sampling', action='store_true', default=False, 
                       help='Disable alias method and use standard weighted sampling')
    parser.add_argument('--enable_false_negative_filter', action='store_true', default=False, 
                       help='Enable false negative filtering (may slow training for large batches)')


    parser.add_argument('--enable_chunked_computation', action='store_true', default=False, help='Enable chunked computation for InfoNCE loss (may reduce memory usage for large batches)')
    parser.add_argument('--log_interval', default=50, type=int, help='Log interval for detailed metrics')

    # Baseline Model construction
    parser.add_argument('--hidden_units', default=256, type=int, help='Transformer hidden units dimension')
    
    # 📏 Embedding维度配置已移至config.py，使用自适应维度分配
    parser.add_argument('--num_blocks', default=24, type=int)
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--num_heads', default=16, type=int)
    parser.add_argument('--dropout_rate', default=0.1, type=float, help='Universal dropout rate for all components')
    parser.add_argument('--emb_dropout_rate', default=0.1, type=float, help='Dropout rate for embeddings and MLPs')
    parser.add_argument('--MLP_dropout_rate', default=0.3, type=float, help='Dropout rate for MLPs (increased for better regularization)')
    parser.add_argument('--id_dropout_rate', default=0.02, type=float, help='Dropout rate for ID embeddings')
    parser.add_argument('--transformer_dropout', default=0.05, type=float, help='Dropout rate for Transformer attention layers (reduced for HSTU)')
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--norm_first', action='store_true', default=True, help='Use pre-norm structure (default: True for better performance)')
    parser.add_argument('--post_norm', dest='norm_first', action='store_false', help='Switch to post-norm structure') # default: False

    # MLP vs DNN选择
    parser.add_argument('--use_enhanced_mlp', action='store_true', default=True, 
                       help='Use enhanced MLP instead of simple DNN for feature processing')
    
    # 🎯 候选侧头部类型选择（seq复杂/cand简单架构优化）
    parser.add_argument('--item_cand_head', default='linear', type=str, 
                       choices=['mlp', 'linear', 'identity', 'light_mlp'],
                       help='候选侧头部类型: mlp(EnhancedDNN), linear(nn.Linear), identity(nn.Identity), light_mlp(轻量MLP)')
    
    # 混合精度训练选项
    parser.add_argument('--enable_mixed_precision', action='store_true', default=False,
                       help='Enable mixed precision training (AMP) for faster GPU training')
    
    # 🔍 数据监控选项
    parser.add_argument('--enable_data_monitoring', action='store_true', default=True,
                       help='启用数据监控，定期检查batch数据的完整性')
    parser.add_argument('--disable_data_monitoring', dest='enable_data_monitoring', action='store_false',
                       help='禁用数据监控')
    parser.add_argument('--monitoring_interval', default=1000, type=int,
                       help='数据监控间隔（每N步打印一次训练数据）')
    parser.add_argument('--monitoring_samples', default=3, type=int,
                       help='每次监控显示的样本数量')
    
    # 优化器选择
    parser.add_argument('--use_muon', action='store_true', default=False,
                       help='使用Muon优化器替代AdamW，适用于Transformer模型')
    parser.add_argument('--muon_lr', default=0.1, type=float,
                       help='Muon优化器学习率（仅用于hidden weights）')
    parser.add_argument('--muon_aux_lr', default=0.02, type=float,
                       help='Muon辅助参数学习率（用于embeddings和gains/biases）')
    
    # InfoNCE loss parameters  
    parser.add_argument('--temperature', default=0.03, type=float, help='Temperature parameter for InfoNCE loss (optimized for stability)')
    parser.add_argument('--infonce_chunk_size', default=2048, type=int, help='Chunk size for InfoNCE loss computation to reduce memory usage')
    
    # === Action-aware margin parameters ===
    parser.add_argument('--enable_action_margin', action='store_true', default=False, help='启用基于动作类型的margin-aware InfoNCE')
    parser.add_argument('--action_margin_click', type=float, default=0.0, help='点击正样本的margin γ(click)')
    parser.add_argument('--action_margin_exposure', type=float, default=0.2, help='曝光正样本的margin γ(exposure)')
    
    # 🎯 CTR特征控制参数
    parser.add_argument('--enable_ctr_feature', action='store_true', default=False,
                       help='Enable CTR (click-through rate) as item static feature (default: True)')
    parser.add_argument('--disable_ctr_feature', dest='enable_ctr_feature', action='store_false',
                       help='Disable CTR feature')

    # MMemb Feature ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], choices=[str(s) for s in range(81, 87)])
    
    # RQ-VAE相关参数
    parser.add_argument('--enable_rqvae', action='store_true', default=False, help='启用RQ-VAE模式')
    parser.add_argument('--disable_rqvae', dest='enable_rqvae', action='store_false', help='禁用RQ-VAE模式')
    parser.add_argument('--use_precomputed_semantic_ids', action='store_true', default=False,
                       help='使用预计算的semantic_id（需要先运行precompute_semantic_ids.py）')
    parser.add_argument('--skip_rqvae_training', action='store_true', default=False,
                       help='跳过RQ-VAE预训练，直接使用现有的semantic id文件')
    parser.add_argument('--rqvae_features', nargs='+', default=['81'], 
                       choices=['81', '82', '83', '84', '85', '86'], help='用于RQ-VAE的特征ID（支持81-86）')
    parser.add_argument('--rqvae_epochs', default=5, type=int, help='RQ-VAE训练轮数')
    parser.add_argument('--rqvae_batch_size', default=1024, type=int, help='RQ-VAE批次大小')
    parser.add_argument('--rqvae_lr', default=0.002, type=float, help='RQ-VAE学习率')
    
    # 时间特征相关参数
    parser.add_argument('--enable_time_features', action='store_true', default=False, 
                       help='启用时间特征（包含时间间隔特征和绝对时间特征）')
    parser.add_argument('--disable_time_features', dest='enable_time_features', action='store_false', 
                       help='禁用时间特征')
    parser.add_argument('--enable_time_bias', action='store_true', default=False,
                       help='启用注意力机制中的时间偏置（time bias），可独立于时间特征使用')
    parser.add_argument('--disable_time_bias', dest='enable_time_bias', action='store_false',
                       help='禁用注意力机制中的时间偏置')
    
    # 🎯 时间差特征隔离选项
    parser.add_argument('--disable_time_diff_features', action='store_true', default=False,
                       help='禁用时间差相关特征，仅保留绝对时间特征（用于测试时间信号冲突）')
    
    # Field-wise投影相关参数已删除
    
    # RoPE位置编码相关参数
    parser.add_argument('--enable_rope', action='store_true', default=False,
                       help='启用RoPE旋转位置编码')
    parser.add_argument('--disable_rope', dest='enable_rope', action='store_false',
                       help='禁用RoPE，使用传统位置编码')
    parser.add_argument('--rope_theta', default=10000.0, type=float,
                       help='RoPE旋转基数')
    parser.add_argument('--rope_max_seq_len', default=512, type=int,
                       help='RoPE最大序列长度')
    
    # 🎯 in-batch负样本相关参数
    parser.add_argument('--enable_inbatch_negatives', action='store_true', default=False,
                       help='启用in-batch负样本')
    parser.add_argument('--disable_inbatch_negatives', dest='enable_inbatch_negatives', action='store_false',
                       help='禁用in-batch负样本')

    # 🎯 HSTU注意力机制相关参数
    parser.add_argument('--attention_mode', default='hstu', choices=['sdpa', 'softmax', 'hstu'],
                       help='注意力计算模式：sdpa(PyTorch SDPA), softmax(标准), hstu(HSTU pointwise SiLU)')
    parser.add_argument('--enable_relative_bias', action='store_true', default=False,
                       help='启用HSTU相对位置偏置(RAB)，与时间偏置结合实现rab_{p,t}')
    parser.add_argument('--disable_relative_bias', dest='enable_relative_bias', action='store_false',
                       help='禁用相对位置偏置，使用传统位置编码或RoPE')
    
    # 🎯 梯度检查点优化参数
    parser.add_argument('--checkpoint_layers', nargs='*', type=int, default=[],
                       help='启用梯度检查点的层索引列表，例如 --checkpoint_layers 0 1 2 表示前3层启用checkpoint')
    parser.add_argument('--enable_projection_checkpoint', action='store_true', default=True,
                       help='启用统一投影层的梯度检查点（默认启用）')
    parser.add_argument('--disable_projection_checkpoint', dest='enable_projection_checkpoint', action='store_false',
                       help='禁用统一投影层的梯度检查点')
    parser.add_argument('--enable_continual_projection_checkpoint', action='store_true', default=True,
                       help='启用连续特征投影层的梯度检查点（默认启用）')
    parser.add_argument('--disable_continual_projection_checkpoint', dest='enable_continual_projection_checkpoint', action='store_false',
                       help='禁用连续特征投影层的梯度检查点')

    args = parser.parse_args()
    
    # 🎯 处理别名采样参数逻辑
    if args.disable_alias_sampling:
        args.enable_alias_sampling = False
        print("🎯 别名采样已禁用，将使用标准加权采样")
    elif args.enable_alias_sampling and not args.enable_popularity_sampling:
        print("⚠️ 警告: --enable_alias_sampling 需要配合 --enable_popularity_sampling 使用")
        print("   自动启用流行度采样")
        args.enable_popularity_sampling = True
    
    
    # 🔧 添加参数一致性检查
    if args.enable_rqvae:
        # 确保rqvae_features和mm_emb_id一致
        rqvae_features_set = set(args.rqvae_features)
        mm_emb_id_set = set(args.mm_emb_id)
        
        if rqvae_features_set != mm_emb_id_set:
            print(f"❌ 参数不一致警告:")
            print(f"   --rqvae_features: {args.rqvae_features}")
            print(f"   --mm_emb_id: {args.mm_emb_id}")
            print(f"🔧 自动修复：统一使用mm_emb_id的值")
            args.rqvae_features = args.mm_emb_id
        
        print(f"✅ RQ-VAE模式启用，使用特征: {args.mm_emb_id}")

    return args


def run_rqvae_pretraining(args):
    """运行RQ-VAE预训练阶段 - 直接调用方式，避免子进程开销"""
    
    print("\n" + "="*60)
    print("开始RQ-VAE预训练阶段")
    print("="*60)
    
    try:
        # 🚀 直接导入和调用，避免子进程开销
        from train_rqvae import train_rqvae_for_feature
        
        print(f"训练参数: features={args.rqvae_features}, epochs={args.rqvae_epochs}, "
              f"batch_size={args.rqvae_batch_size}, lr={args.rqvae_lr}, "
              f"mixed_precision={args.enable_mixed_precision}")
        
        # 为每个特征依次训练RQ-VAE
        for feature_id in args.rqvae_features:
            print(f"\n🎯 开始训练特征 {feature_id} 的 RQ-VAE")
            train_rqvae_for_feature(
                feature_id=feature_id,
                device=args.device,
                epochs=args.rqvae_epochs,
                batch_size=args.rqvae_batch_size,
                lr=args.rqvae_lr,
                enable_mixed_precision=args.enable_mixed_precision
            )
            print(f"✅ 特征 {feature_id} 的 RQ-VAE 训练完成")
        
        print("🎉 RQ-VAE预训练全部完成!")
        return True
        
    except KeyboardInterrupt:
        print("⚠️ RQ-VAE预训练被手动中断")
        return False
    except Exception as e:
        print(f"❌ RQ-VAE预训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    Path(os.environ.get('TRAIN_LOG_PATH')).mkdir(parents=True, exist_ok=True)
    Path(os.environ.get('TRAIN_TF_EVENTS_PATH')).mkdir(parents=True, exist_ok=True)
    log_file = open(Path(os.environ.get('TRAIN_LOG_PATH'), 'train.log'), 'w')
    writer = SummaryWriter(os.environ.get('TRAIN_TF_EVENTS_PATH'))
    data_path = os.environ.get('TRAIN_DATA_PATH')

    args = get_args()
    
    # RQ-VAE预训练阶段
    if args.enable_rqvae:
        if args.skip_rqvae_training:
            print("\n" + "="*60)
            print("跳过RQ-VAE预训练")
            print("="*60)
        else:
            success = run_rqvae_pretraining(args)
            if not success:
                print("RQ-VAE预训练失败，将继续进行baseline训练...")
            else:
                print("RQ-VAE预训练完成，semantic id特征将在baseline训练中自动加载")
    
    print("\n" + "="*60)
    print("开始Baseline模型训练阶段")
    print("="*60)
    
    dataset = MyDataset(data_path, args)
    
    # 🎯 全量训练 vs 验证集划分
    if args.full_train:
        print("🚀 全量训练模式：使用所有数据进行训练，trust public leaderboard")
        train_dataset = dataset
        valid_dataset = None
        valid_loader = None
    else:
        print(f"📊 验证集划分模式：训练集 {1-args.valid_ratio:.1%}，验证集 {args.valid_ratio:.1%}")
        train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [1-args.valid_ratio, args.valid_ratio])
        
        # 创建验证集DataLoader
        valid_loader = DataLoader(
            valid_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=4, 
            collate_fn=dataset.collate_fn, 
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=2
        )
    
    # ⚡ 多进程优化：大幅提升数据加载速度
    num_workers = min(4, os.cpu_count() or 1)  # 提升到8个worker
    print(f"🚀 启用DataLoader多进程加速: num_workers={num_workers}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        collate_fn=dataset.collate_fn, 
        persistent_workers=True,
        pin_memory=True,  # GPU加速
        prefetch_factor=2  # 预取因子
    )
    usernum, itemnum = dataset.usernum, dataset.itemnum # 表示用户和物品的数量
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types #表示特征的统计信息和类型
    
    # 🎯 如果是从global_step格式的路径继续训练，重新计算epoch_start_idx
    # (这个计算已经移动到global_step初始化部分)

    # 简化dropout参数设置 - 统一使用经验最佳值
    if not hasattr(args, 'emb_dropout_rate'):
        args.emb_dropout_rate = args.dropout_rate
        print(f"emb_dropout_rate设置：emb_dropout_rate={args.dropout_rate} ")
    if not hasattr(args, 'MLP_dropout_rate'):
        args.MLP_dropout_rate = args.dropout_rate
        print(f"MLP_dropout_rate设置：MLP_dropout_rate={args.dropout_rate} ")
    if not hasattr(args, 'transformer_dropout'):
        args.transformer_dropout = args.dropout_rate
        print(f"transformer_dropout设置：transformer_dropout={args.dropout_rate} ")
    
    # 🎯 处理自动checkpoint配置
    if hasattr(args, '_auto_checkpoint_first_half') and args._auto_checkpoint_first_half:
        first_half_count = args.num_blocks // 2
        args.checkpoint_layers = list(range(first_half_count))
        print(f"🎯 自动生成前{first_half_count}层的梯度检查点: {args.checkpoint_layers}")

    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)

    epoch_start_idx = 1
    global_step_to_resume = None  # 初始化变量

    if args.state_dict_path is not None: #  若提供了 --state_dict_path，尝试载入权重；并从路径中解析 epoch= 或 global_step 后的数字，设置为下一训练起点（断点续训）。
        try:
            # 🔍 智能路径查找：优先从cache目录读取，如果没有找到再从原路径读取
            actual_model_path = args.state_dict_path
            
            # 检查是否需要从cache目录查找模型
            if not os.path.exists(args.state_dict_path):
                # 获取cache目录路径
                cache_dir = os.environ.get('USER_CACHE_PATH')
                if cache_dir is None:
                    cache_dir = str(Path.home() / "user_cache")
                
                # 从state_dict_path中提取目录名和文件名信息
                import re
                path_str = str(args.state_dict_path)
                
                # 尝试匹配不同的路径格式
                if 'global_step' in path_str:
                    # 匹配 global_stepXXX.valid_loss=X.XXXX 格式
                    match = re.search(r'global_step(\d+)(?:\.valid_loss=([\d\.]+))?', path_str)
                    if match:
                        global_step_num = match.group(1)
                        valid_loss = match.group(2)
                        if valid_loss:
                            cache_dir_name = f"global_step{global_step_num}.valid_loss={valid_loss}"
                        else:
                            cache_dir_name = f"global_step{global_step_num}"
                        cache_model_path = Path(cache_dir) / cache_dir_name / "model.pt"
                        if cache_model_path.exists():
                            actual_model_path = str(cache_model_path)
                            print(f"🔄 从cache目录找到模型: {actual_model_path}")
                elif 'epoch' in path_str:
                    # 匹配 epochXXX 格式
                    match = re.search(r'epoch(\d+)', path_str)
                    if match:
                        epoch_num = match.group(1)
                        cache_dir_name = f"epoch{epoch_num}.global_step0"
                        cache_model_path = Path(cache_dir) / cache_dir_name / "model.pt"
                        if cache_model_path.exists():
                            actual_model_path = str(cache_model_path)
                            print(f"🔄 从cache目录找到模型: {actual_model_path}")
                
                # 如果cache中也没有找到，提示用户
                if actual_model_path == args.state_dict_path:
                    print(f"⚠️ 原路径不存在，cache目录中也未找到对应模型: {args.state_dict_path}")
                    print(f"   Cache目录: {cache_dir}")
            
            model.load_state_dict(torch.load(actual_model_path, map_location=torch.device(args.device)))
            
            # 支持两种路径格式：epoch=N 和 global_stepN（使用实际加载的路径进行解析）
            path_for_parsing = actual_model_path
            if 'epoch=' in path_for_parsing:
                # 原有的epoch格式：epoch=5.pt -> epoch_start_idx = 6
                tail = path_for_parsing[path_for_parsing.find('epoch=') + 6 :]
                epoch_start_idx = int(tail[: tail.find('.')]) + 1
            elif 'global_step' in path_for_parsing:
                # 新的global_step格式：global_step3875.valid_loss=7.8523 -> 根据global_step推算epoch
                import re
                match = re.search(r'global_step(\d+)', path_for_parsing)
                if match:
                    global_step_to_resume = int(match.group(1))
                    # 根据每个epoch的步数估算当前应该开始的epoch
                    # 这里假设每个epoch大约有len(train_loader)步，但由于train_loader还未创建，
                    # 我们需要在DataLoader创建后重新计算
                    print(f'从global_step {global_step_to_resume} 继续训练，将在DataLoader创建后重新计算epoch_start_idx')
                    # 暂时设置一个标记，稍后重新计算
                    epoch_start_idx = -1  # 特殊标记，表示需要重新计算
                else:
                    raise ValueError(f"无法从路径中解析global_step: {path_for_parsing}")
            else:
                raise ValueError(f"路径格式不支持，需要包含'epoch='或'global_step': {path_for_parsing}")
                
            print(f'成功加载权重: {actual_model_path}')
        except Exception as e:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            print(f'Error: {e}')
            raise RuntimeError('failed loading state_dicts, pls check file path!')

    # 优化器选择：Muon vs AdamW
    if args.use_muon:
        try:
            from muon import MuonWithAuxAdam
            import torch.distributed as dist
            if not dist.is_initialized():
                try:
                    import tempfile
                    tmp_init_file = tempfile.NamedTemporaryFile(delete=False)
                    dist.init_process_group(
                        backend="gloo",
                        init_method=f"file://{tmp_init_file.name}",
                        rank=0,
                        world_size=1,
                    )
                except Exception as _file_init_exc:
                    print(f"⚠️ file:// 初始化失败 ({_file_init_exc}), 回退到 env://")
                    # 确保必要环境变量存在
                    os.environ.setdefault("MASTER_ADDR", "localhost")
                    os.environ.setdefault("MASTER_PORT", "12355")
                    dist.init_process_group(
                        backend="gloo",
                        init_method="env://",
                        rank=0,
                        world_size=1,
                    )
            
            # 按照Muon最佳实践分组参数
            # 1. hidden_weights: 高维参数（≥2维），使用Muon
            # 2. gains_biases: 低维参数（<2维），使用Adam
            # 3. embeddings: 嵌入层参数，使用Adam
            
            hidden_weights = []
            gains_biases = []
            embedding_params = []
            semantic_embedding_params = []  # 🔥 新增：semantic embedding专用参数组
            
            # 分类模型参数 - 只包含需要梯度的参数，排除冻结参数
            for name, param in model.named_parameters():
                # 跳过冻结参数（如RQVAE相关参数）和不需要梯度的参数
                if not param.requires_grad:
                    print(f"跳过冻结参数: {name}")
                    continue
                    
                # 跳过RQVAE相关的参数（这些通常是冻结的）
                if 'rqvae' in name.lower():
                    print(f"跳过RQVAE参数: {name}")
                    continue
                
                # 🔥 semantic embedding参数单独分组（用于复用codebook权重的微调）
                if 'semantic_embeds' in name or 'semantic_layer_weights' in name or 'semantic_norms' in name:
                    semantic_embedding_params.append(param)
                    print(f"🎯 Semantic参数: {name} -> 独立参数组")
                elif 'emb' in name or 'embedding' in name:  # 其他嵌入层参数
                    embedding_params.append(param)
                elif param.ndim >= 2:  # 高维参数（权重矩阵）
                    hidden_weights.append(param)
                else:  # 低维参数（偏置、LayerNorm等）
                    gains_biases.append(param)
            
            # 确保每个参数组都有参数
            if len(hidden_weights) == 0:
                print("⚠️ 警告: 没有找到高维参数，Muon可能不适用")
            if len(gains_biases) == 0 and len(embedding_params) == 0 and len(semantic_embedding_params) == 0:
                print("⚠️ 警告: 没有找到低维/嵌入参数")
            
            # 构建参数组 - 只有当参数组非空时才添加
            param_groups = []
            if len(hidden_weights) > 0:
                param_groups.append(dict(
                    params=hidden_weights, 
                    use_muon=True,
                    lr=args.muon_lr, 
                    weight_decay=args.weight_decay  # 仅对高维权重做衰减
                ))
            
            # 将 gains_biases 与 embedding_params 拆分，二者都不做 weight decay
            if len(gains_biases) > 0:
                param_groups.append(dict(
                    params=gains_biases, 
                    use_muon=False,
                    lr=args.muon_aux_lr, 
                    betas=(0.9, 0.98), 
                    weight_decay=0.0
                ))
            if len(embedding_params) > 0:
                param_groups.append(dict(
                    params=embedding_params, 
                    use_muon=False,
                    lr=args.muon_aux_lr, 
                    betas=(0.9, 0.98), 
                    weight_decay=0.0
                ))
            # 🔥 semantic embedding参数组：使用更小学习率和L2正则（用于codebook微调）
            if len(semantic_embedding_params) > 0:
                from config import get_semantic_id_config
                semantic_config = get_semantic_id_config()
                global_config = semantic_config.get('rqvae_alignment', {})
                semantic_lr = args.muon_aux_lr * global_config.get('precompute_fine_tune_lr_ratio', 0.1)
                semantic_wd = global_config.get('regularization_weight', 1e-4) if global_config.get('enable_regularization', True) else 0.0
                param_groups.append(dict(
                    params=semantic_embedding_params,
                    use_muon=False,
                    lr=semantic_lr,
                    betas=(0.9, 0.98),
                    weight_decay=semantic_wd
                ))
                print(f"🎯 Semantic参数组: lr={semantic_lr:.6f} (ratio={global_config.get('precompute_fine_tune_lr_ratio', 0.1)}), wd={semantic_wd}")
            
            if len(param_groups) == 0:
                raise ValueError("没有找到任何可训练参数，无法创建优化器")
            
            optimizer = MuonWithAuxAdam(param_groups)
            total_aux_params = len(gains_biases + embedding_params + semantic_embedding_params)
            print(f"🚀 使用Muon优化器: hidden_weights={len(hidden_weights)}, aux_params={total_aux_params}")
            print(f"   - Hidden: {len(hidden_weights)}, Gains/Biases: {len(gains_biases)}, Embeddings: {len(embedding_params)}, Semantic: {len(semantic_embedding_params)}")
            print(f"   - Muon LR: {args.muon_lr}, Aux LR: {args.muon_aux_lr}")
            
        except ImportError:
            print("❌ 无法导入Muon优化器，回退到AdamW")
            print("   请安装Muon: pip install muon-optimizer")
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=args.weight_decay)
        except Exception as e:
            print(f"❌ Muon优化器初始化失败: {e}")
            print("   回退到AdamW优化器")
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=args.weight_decay)
    else:
        # 使用传统AdamW优化器（同样进行参数分组，避免对 Embedding/bias/LayerNorm 做衰减）
        hidden_weights = []
        gains_biases = []
        embedding_params = []
        semantic_embedding_params = []  # 🔥 新增：semantic embedding专用参数组
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'rqvae' in name.lower():
                continue
            # 🔥 semantic embedding参数单独分组（用于复用codebook权重的微调）
            if 'semantic_embeds' in name or 'semantic_layer_weights' in name or 'semantic_norms' in name:
                semantic_embedding_params.append(param)
            elif 'emb' in name or 'embedding' in name:
                embedding_params.append(param)
            elif param.ndim >= 2:
                hidden_weights.append(param)
            else:
                gains_biases.append(param)
        
        adamw_param_groups = []
        if len(hidden_weights) > 0:
            adamw_param_groups.append({
                'params': hidden_weights,
                'lr': args.lr,
                'betas': (0.9, 0.98),
                'weight_decay': args.weight_decay  # 仅高维权重衰减
            })
        if len(gains_biases) > 0:
            adamw_param_groups.append({
                'params': gains_biases,
                'lr': args.lr,
                'betas': (0.9, 0.98),
                'weight_decay': 0.0
            })
        if len(embedding_params) > 0:
            adamw_param_groups.append({
                'params': embedding_params,
                'lr': args.lr,
                'betas': (0.9, 0.98),
                'weight_decay': 0.0
            })
        # 🔥 semantic embedding参数组：使用更小学习率和L2正则（用于codebook微调）
        if len(semantic_embedding_params) > 0:
            from config import get_semantic_id_config
            semantic_config = get_semantic_id_config()
            global_config = semantic_config.get('rqvae_alignment', {})
            semantic_lr = args.lr * global_config.get('precompute_fine_tune_lr_ratio', 0.1)
            semantic_wd = global_config.get('regularization_weight', 1e-4) if global_config.get('enable_regularization', True) else 0.0
            adamw_param_groups.append({
                'params': semantic_embedding_params,
                'lr': semantic_lr,
                'betas': (0.9, 0.98),
                'weight_decay': semantic_wd
            })
            print(f"🎯 Semantic参数组: lr={semantic_lr:.6f} (ratio={global_config.get('precompute_fine_tune_lr_ratio', 0.1)}), wd={semantic_wd}")
        
        optimizer = torch.optim.AdamW(adamw_param_groups)
        print(f"📊 使用AdamW优化器(分组): lr={args.lr}, wd(hidden)={args.weight_decay}, wd(emb/bias/norm)=0")
        print(f"   - Hidden: {len(hidden_weights)}, Gains/Biases: {len(gains_biases)}, Embeddings: {len(embedding_params)}, Semantic: {len(semantic_embedding_params)}")
    
    # Calculate total training steps for better scheduling
    total_steps = len(train_loader) * args.num_epochs
    
    # Advanced learning rate scheduler with multiple strategies
    def get_advanced_lr_scale_factor(step):
        if step < args.warmup_steps:
            # Linear warmup
            return float(step) / float(max(1, args.warmup_steps))
        
        # Post-warmup scheduling
        post_warmup_steps = step - args.warmup_steps
        remaining_steps = total_steps - args.warmup_steps
        
        if args.lr_schedule == 'constant':
            return 1.0
        elif args.lr_schedule == 'cosine':
            # Cosine annealing
            return args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (
                1 + np.cos(np.pi * post_warmup_steps / remaining_steps)
            )
        elif args.lr_schedule == 'cosine_with_restarts':
            # Cosine annealing with warm restarts (period = remaining_steps / 2)
            restart_period = max(remaining_steps // 2, 1)
            t_cur = post_warmup_steps % restart_period
            return args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (
                1 + np.cos(np.pi * t_cur / restart_period)
            )
        elif args.lr_schedule == 'polynomial':
            # Polynomial decay
            return args.min_lr_ratio + (1 - args.min_lr_ratio) * (
                1 - post_warmup_steps / remaining_steps
            ) ** 2
        else:
            return 1.0
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_advanced_lr_scale_factor)
    
    # 🔧 恢复学习率调度器状态（如果从checkpoint继续训练）
    if global_step_to_resume is not None and global_step_to_resume > 0:
        # 手动设置调度器的last_epoch，使其从正确的步数开始
        scheduler.last_epoch = global_step_to_resume - 1  # last_epoch是0-indexed
        # 更新学习率到正确的值
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr'] if hasattr(optimizer, 'param_groups') else args.lr
        print(f'🔧 学习率调度器状态已恢复: step={global_step_to_resume}, lr={current_lr:.6f}')
        
        # 检查是否已过warmup阶段
        if global_step_to_resume >= args.warmup_steps:
            print(f'✅ 已跳过warmup阶段 (warmup_steps={args.warmup_steps})')
        else:
            remaining_warmup = args.warmup_steps - global_step_to_resume
            print(f'⏳ 仍在warmup阶段，剩余{remaining_warmup}步完成warmup')
    
    # 🚀 混合精度训练支持 - 升级到BF16提升数值稳定性
    scaler = None
    mixed_precision_dtype = None
    if args.enable_mixed_precision and torch.cuda.is_available():
        # 检查BF16支持情况
        if torch.cuda.is_bf16_supported():
            mixed_precision_dtype = torch.bfloat16
            # BF16不需要GradScaler，数值稳定性更好
            scaler = None
            print("🚀 启用BF16混合精度训练 - 更好的数值稳定性和性能（无需GradScaler）")
        else:
            # 回退到FP16，需要GradScaler
            mixed_precision_dtype = torch.float16
            scaler = torch.cuda.amp.GradScaler()
            print("⚠️ 当前GPU不支持BF16，回退到FP16混合精度训练（使用GradScaler）")
    elif args.enable_mixed_precision:
        print("⚠️ 混合精度训练需要CUDA支持，已禁用")
    else:
        print("📊 使用传统FP32精度训练")
    
    # Helper functions for monitoring and adaptive clipping
    def compute_grad_norm(parameters):
        total_norm = 0.0
        max_grad = 0.0
        param_count = 0
        nan_count = 0
        inf_count = 0
        
        for p in parameters:
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                max_grad = max(max_grad, p.grad.data.abs().max().item())
                param_count += 1
                
                if torch.isnan(p.grad.data).any():
                    nan_count += 1
                if torch.isinf(p.grad.data).any():
                    inf_count += 1
        
        return {
            'total_norm': total_norm ** 0.5,
            'max_grad': max_grad,
            'nan_params': nan_count,
            'inf_params': inf_count,
            'total_params_with_grad': param_count,
            'has_nan': nan_count > 0,
            'has_inf': inf_count > 0
        }
    
    # 高效的自适应梯度裁剪
    if args.adaptive_grad_clip:
        gradient_history = np.zeros(100, dtype=np.float32)  # 固定大小的环形缓冲区
        grad_history_idx = 0
        grad_history_full = False
        
        # 🔧 如果从checkpoint恢复，重置自适应梯度裁剪历史
        if global_step_to_resume is not None and global_step_to_resume > 0:
            print(f'🔧 自适应梯度裁剪历史已重置（从checkpoint恢复）')
    adaptive_clip_factor = args.max_grad_norm
    
    def compute_param_norm(parameters):
        total_norm = 0.0
        for p in parameters:
            param_norm = p.data.norm(2)
            total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def fill_none_grads(optim, log_stats=False, writer=None, step=None):
        """将所有 None gradients 替换为零梯度（in-place）以兼容 Muon。"""
        none_count = 0
        total_params = 0
        
        for _group in optim.param_groups:
            for _p in _group["params"]:
                total_params += 1
                if _p.grad is None:
                    _p.grad = torch.zeros_like(_p.data)
                    none_count += 1
        
        # 🎯 记录None梯度统计（仅在使用Muon时）
        # if log_stats and args.use_muon and writer is not None and step is not None:
        #     writer.add_scalar('Optimizer/none_grads_filled', none_count, step)
        #     writer.add_scalar('Optimizer/none_grad_ratio', none_count / max(total_params, 1), step)
            
        #     # 如果None梯度比例过高，记录警告
        #     if none_count / max(total_params, 1) > 0.1:  # 超过10%
        #         writer.add_scalar('Optimizer/high_none_grad_warning', 1.0, step)
        
        return none_count, total_params

    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0
    T = 0.0
    t0 = time.time()    
    global_step = 0
    
    # 🎯 如果是从global_step格式的路径继续训练，重新计算epoch_start_idx和global_step
    if global_step_to_resume is not None:
        steps_per_epoch = len(train_loader)
        completed_epochs = global_step_to_resume // steps_per_epoch
        epoch_start_idx = completed_epochs + 1
        global_step = global_step_to_resume  # 从保存的global_step继续
        print(f'📊 从global_step {global_step_to_resume} 继续训练：')
        print(f'   - 每epoch步数: {steps_per_epoch}')
        print(f'   - 已完成epoch: {completed_epochs}')
        print(f'   - 开始epoch: {epoch_start_idx}')
        print(f'   - 继续global_step: {global_step}')
    
    # 计算总训练步数，用于动态exposure_weight调整
    total_steps = len(train_loader) * args.num_epochs
    
    # 🎯 记录优化器配置到TensorBoard
    optimizer_type = "Muon" if args.use_muon else "AdamW"
    writer.add_text('Config/optimizer_type', optimizer_type, 0)
    
    # if args.use_muon and hasattr(optimizer, 'param_groups'):
    #     # 记录Muon优化器的详细配置
    #     for i, param_group in enumerate(optimizer.param_groups):
    #         use_muon = param_group.get('use_muon', False)
    #         group_type = 'Muon' if use_muon else 'Adam'
    #         param_count = len(param_group['params'])
    #         lr = param_group['lr']
            
    #         writer.add_text(f'Config/{group_type}_group_{i}', 
    #                       f'lr={lr}, param_count={param_count}', 0)
        
    #     # 记录总体配置
    #     total_muon_params = sum(len(g['params']) for g in optimizer.param_groups if g.get('use_muon', False))
    #     total_adam_params = sum(len(g['params']) for g in optimizer.param_groups if not g.get('use_muon', False))
        
    #     writer.add_scalar('Config/total_muon_params', total_muon_params, 0)
    #     writer.add_scalar('Config/total_adam_params', total_adam_params, 0)
        
    #     print(f"🎯 Muon优化器配置已记录到TensorBoard: Muon参数={total_muon_params}, Adam参数={total_adam_params}")
    # else:
    #     writer.add_text('Config/optimizer_details', f'AdamW with lr={args.lr}', 0)
    #     print(f"📊 AdamW优化器配置已记录到TensorBoard")
    
    # 打印动态调整信息
    if args.exposure_weight_start is not None and args.exposure_weight_end is not None:
        print(f"动态exposure_weight: {args.exposure_weight_start:.2f} -> {args.exposure_weight_end:.2f} ({args.exposure_decay_strategy})")
    else:
        print(f"固定exposure_weight: {args.exposure_weight:.2f}")
    
    print("Start training")
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        model.train()
        if args.inference_only:
            break
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            # ⚡ 新的数据格式：直接从Dataset获取tensor字典
            optimizer.zero_grad()
            # optimizer.zero_grad(set_to_none=True)
            # 🚀 混合精度训练：autocast上下文管理器（BF16优先）
            if scaler is not None:
                with torch.cuda.amp.autocast(dtype=mixed_precision_dtype):
                    # Model直接接收batch字典，无需手动解包
                    seq_embs, pos_embs, neg_embs, loss_mask, pos_seqs, neg_seqs, next_action_type_out = model(batch)
                    # 计算当前步的动态exposure_weight
                    current_exposure_weight = get_dynamic_exposure_weight(args, global_step, total_steps)
                    # 推荐系统专用：核心改进 - 曝光样本加权 + false negative过滤
                    pos_seqs_for_filter = pos_seqs if args.enable_false_negative_filter else None
                    neg_seqs_for_filter = neg_seqs if args.enable_false_negative_filter else None
                    loss = model.compute_infonce_loss(
                        seq_embs, pos_embs, neg_embs, loss_mask, 
                        pos_seqs=pos_seqs_for_filter, neg_seqs=neg_seqs_for_filter, next_action_type=next_action_type_out,
                        exposure_weight=current_exposure_weight, writer=writer, global_step=global_step, enable_chunked_computation=args.enable_chunked_computation, chunk_size=args.infonce_chunk_size
                    )
            else:
                # 传统FP32训练
                seq_embs, pos_embs, neg_embs, loss_mask, pos_seqs, neg_seqs, next_action_type_out = model(batch)
                # 计算当前步的动态exposure_weight
                current_exposure_weight = get_dynamic_exposure_weight(args, global_step, total_steps)
                # 推荐系统专用：核心改进 - 曝光样本加权 + false negative过滤
                pos_seqs_for_filter = pos_seqs if args.enable_false_negative_filter else None
                neg_seqs_for_filter = neg_seqs if args.enable_false_negative_filter else None
                loss = model.compute_infonce_loss(
                        seq_embs, pos_embs, neg_embs, loss_mask, 
                        pos_seqs=pos_seqs_for_filter, neg_seqs=neg_seqs_for_filter, next_action_type=next_action_type_out,
                        exposure_weight=current_exposure_weight, writer=writer, global_step=global_step, enable_chunked_computation=args.enable_chunked_computation, chunk_size=args.infonce_chunk_size
                    )

            # 🎯 增强的训练日志 - 包含优化器信息
            optimizer_type = "Muon" if args.use_muon else "AdamW"
            current_lr = optimizer.param_groups[0]['lr'] if hasattr(optimizer, 'param_groups') else args.lr
            
            log_json = json.dumps({
                'global_step': global_step, 
                'loss': loss.item(), 
                'epoch': epoch, 
                'exposure_weight': current_exposure_weight,
                'optimizer': optimizer_type,
                'lr': current_lr,
                'time': time.time()
            })
            log_file.write(log_json + '\n')
            log_file.flush()
            print(log_json)

            writer.add_scalar('Loss/train', loss.item(), global_step)
            writer.add_scalar('Train/exposure_weight', current_exposure_weight, global_step)
            writer.add_scalar('Train/current_lr', current_lr, global_step)

            # 修正 Embedding 正则：使用"平方 L2"，且排除 padding 行
            if args.l2_emb > 0:
                # item embedding
                if hasattr(model, 'item_emb'):
                    emb = model.item_emb.weight
                    pad_idx = getattr(model.item_emb, 'padding_idx', None)
                    if pad_idx is not None:
                        emb_no_pad = torch.cat([emb[:pad_idx], emb[pad_idx+1:]], dim=0)
                    else:
                        emb_no_pad = emb
                    loss += args.l2_emb * (emb_no_pad.pow(2).sum() / emb_no_pad.numel())

                # user embedding
                if hasattr(model, 'user_emb'):
                    emb = model.user_emb.weight
                    pad_idx = getattr(model.user_emb, 'padding_idx', None)
                    if pad_idx is not None:
                        emb_no_pad = torch.cat([emb[:pad_idx], emb[pad_idx+1:]], dim=0)
                    else:
                        emb_no_pad = emb
                    loss += args.l2_emb * (emb_no_pad.pow(2).sum() / emb_no_pad.numel())
            
            # 🎯 新增：语义ID相关的正则化损失
            # if args.enable_rqvae:
            #     semantic_reg_loss = model.compute_semantic_regularization_loss()
            #     if semantic_reg_loss.item() > 0:
            #         loss += semantic_reg_loss
            #         writer.add_scalar('Loss/semantic_regularization', semantic_reg_loss.item(), global_step)
            
            # 🚀 混合精度训练：反向传播
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 🔍 NaN梯度检测和跳过机制（仅混合精度训练）
            if scaler is not None:  # 仅混合精度训练时检测
                grad_stats = compute_grad_norm(model.parameters())
                if grad_stats['has_nan'] or grad_stats['has_inf']:
                    print(f"⚠️ 检测到NaN/Inf梯度，跳过第{global_step}步更新")
                    writer.add_scalar('Train/skipped_steps_nan', 1, global_step)
                    # 清除梯度并跳过优化器更新
                    optimizer.zero_grad()
                    # optimizer.zero_grad(set_to_none=True)
                    continue  # 跳过当前batch
      
            # Advanced gradient clipping with adaptive strategy
            if args.max_grad_norm > 0:
                # Get current gradient norm for adaptive clipping
                if args.adaptive_grad_clip:
                    current_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
                    # 高效的自适应梯度裁剪 - 使用环形缓冲区
                    gradient_history[grad_history_idx] = current_grad_norm.item()
                    grad_history_idx = (grad_history_idx + 1) % 100
                    if grad_history_idx == 0:
                        grad_history_full = True
                    
                    # 只有积累了足够历史时才计算自适应阈值
                    if grad_history_full or grad_history_idx >= 10:
                        valid_history = gradient_history[:grad_history_idx] if not grad_history_full else gradient_history
                        percentile_90 = np.percentile(valid_history, 90)
                        adaptive_clip_value = min(args.max_grad_norm * 2.0, percentile_90 * 1.5)
                        # 平滑过渡
                        adaptive_clip_factor = 0.9 * adaptive_clip_factor + 0.1 * adaptive_clip_value
                    
                    # Apply adaptive clipping
                    grad_norm_clipped = torch.nn.utils.clip_grad_norm_(model.parameters(), adaptive_clip_factor)
                    clip_value_used = adaptive_clip_factor
                else:
                    # Standard fixed clipping
                    grad_norm_clipped = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    clip_value_used = args.max_grad_norm
                
                # Only log clipping stats when logging
                if global_step % args.log_interval == 0:
                    clip_ratio = min(1.0, clip_value_used / (grad_norm_clipped.item() + 1e-8))
                    writer.add_scalar('Train/grad_clip_ratio', clip_ratio, global_step)
            
            # 混合精度训练：优化器更新
            log_none_grads = (global_step % args.log_interval == 0)  # 只在记录间隔时统计
            
            if scaler is not None:
                none_count, total_params = fill_none_grads(optimizer, log_none_grads, writer, global_step)
                scaler.step(optimizer)
                scaler.update()
            else:
                none_count, total_params = fill_none_grads(optimizer, log_none_grads, writer, global_step)
                optimizer.step()

            # 重要：重置各 Embedding 的 padding 行，防止 decoupled weight decay 使其漂移
            with torch.no_grad():
                def zero_pad_row(emb_mod):
                    if hasattr(emb_mod, 'padding_idx') and emb_mod.padding_idx is not None:
                        emb_mod.weight.data[emb_mod.padding_idx].zero_()
                if hasattr(model, 'item_emb'):
                    zero_pad_row(model.item_emb)
                if hasattr(model, 'user_emb'):
                    zero_pad_row(model.user_emb)
                if hasattr(model, 'pos_emb'):
                    zero_pad_row(model.pos_emb)

            scheduler.step()
            global_step += 1
            
            # 🎯 全量训练模式：每隔1/8个epoch保存一次模型
            
            steps_per_eighth_epoch = len(train_loader) // 8
            if steps_per_eighth_epoch > 0 and global_step % steps_per_eighth_epoch == 0:
                print(f"Global step {global_step} 达到1/8 epoch间隔，保存模型。")
                save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), f"global_step{global_step}.epoch{epoch}")
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_dir / "model.pt")

        # 验证阶段：根据训练模式决定是否进行验证
        if args.full_train:
            # 全量训练模式：跳过验证，epoch结束时不再保存（已在训练过程中保存）
            print(f"Epoch {epoch} 完成，全量训练模式跳过验证。")
            pass  # 不再在epoch结束时保存
        else:
            # 验证集模式：计算验证损失
            model.eval()
            valid_loss_sum = 0
            valid_click_loss_sum = 0
            valid_expo_loss_sum = 0
            valid_click_pos_sum = 0
            valid_click_neg_sum = 0
            
            for step, batch in tqdm(enumerate(valid_loader), total=len(valid_loader)):
                with torch.no_grad():
                    if scaler is not None:  # 混合精度模式（BF16优先）
                        with torch.cuda.amp.autocast(dtype=mixed_precision_dtype):
                            seq_embs, pos_embs, neg_embs, loss_mask, pos_seqs, neg_seqs, next_action_type_out = model(batch)
                    else:  # FP32模式
                        seq_embs, pos_embs, neg_embs, loss_mask, pos_seqs, neg_seqs, next_action_type_out = model(batch)
                    # 在验证时也使用动态exposure_weight保持一致性
                    current_exposure_weight = get_dynamic_exposure_weight(args, global_step, total_steps)
                    pos_seqs_for_filter = pos_seqs if args.enable_false_negative_filter else None
                    neg_seqs_for_filter = neg_seqs if args.enable_false_negative_filter else None
                    
                    # 获取详细的损失分解
                    loss, click_loss, expo_loss = model.compute_infonce_loss(
                        seq_embs, pos_embs, neg_embs, loss_mask, 
                        pos_seqs=pos_seqs_for_filter, neg_seqs=neg_seqs_for_filter, next_action_type=next_action_type_out,
                        exposure_weight=current_exposure_weight, writer=writer, global_step=global_step, 
                        enable_chunked_computation=args.enable_chunked_computation, chunk_size=args.infonce_chunk_size,
                        return_detailed_loss=True
                    )
                    
                    # 计算正样本和负样本的相似度（用于valid_click_pos和valid_click_neg）
                    # 先进行L2归一化，与compute_infonce_loss中的处理保持一致
                    import torch.nn.functional as F
                    eps = 1e-5
                    seq_embs_norm = F.normalize(seq_embs, p=2, dim=-1, eps=eps)
                    pos_embs_norm = F.normalize(pos_embs, p=2, dim=-1, eps=eps)
                    neg_embs_norm = F.normalize(neg_embs, p=2, dim=-1, eps=eps)
                    
                    # 计算正样本相似度（点积）
                    pos_sim = torch.sum(seq_embs_norm * pos_embs_norm, dim=-1)  # [batch_size, maxlen]
                    # 计算负样本相似度（取平均）
                    neg_sim = torch.sum(seq_embs_norm * neg_embs_norm, dim=-1)  # [batch_size, maxlen]
                    
                    # 只在有效位置计算平均相似度
                    if loss_mask.any():
                        valid_pos_sim = pos_sim[loss_mask.bool()].mean().item()
                        valid_neg_sim = neg_sim[loss_mask.bool()].mean().item()
                    else:
                        valid_pos_sim = 0.0
                        valid_neg_sim = 0.0
                    
                    valid_loss_sum += loss.item()
                    valid_click_loss_sum += click_loss.item()
                    valid_expo_loss_sum += expo_loss.item()
                    valid_click_pos_sum += valid_pos_sim
                    valid_click_neg_sum += valid_neg_sim
                    
            with torch.no_grad():
                # 计算平均指标
                valid_loss_sum /= len(valid_loader)
                valid_click_loss_sum /= len(valid_loader)
                valid_expo_loss_sum /= len(valid_loader)
                valid_click_pos_sum /= len(valid_loader)
                valid_click_neg_sum /= len(valid_loader)
            
            # 记录所有指标
            writer.add_scalar('Loss/valid', valid_loss_sum, global_step)
            writer.add_scalar('Loss/valid_click', valid_click_loss_sum, global_step)
            writer.add_scalar('Loss/valid_expo', valid_expo_loss_sum, global_step)
            writer.add_scalar('Loss/valid_click_pos', valid_click_pos_sum, global_step)
            writer.add_scalar('Loss/valid_click_neg', valid_click_neg_sum, global_step)
            
            print(f"Validation - Loss: {valid_loss_sum:.4f}, Click Loss: {valid_click_loss_sum:.4f}, Expo Loss: {valid_expo_loss_sum:.4f}, Click Pos: {valid_click_pos_sum:.4f}, Click Neg: {valid_click_neg_sum:.4f}")
            save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), f"global_step{global_step}.valid_loss={valid_loss_sum:.4f}")
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_dir / "model.pt")

    print("Done")
    
    # 🎯 训练结束统计
    final_optimizer_type = "Muon" if args.use_muon else "AdamW"
    print(f"🎯 训练完成！最终使用的优化器: {final_optimizer_type}")
    
    if args.use_muon:
        print("📊 Muon优化器使用总结:")
        print("   - 检查TensorBoard中的 'Optimizer/*' 指标以查看详细统计")
        print("   - 特别关注None梯度填充统计和参数组学习率变化")
    
    # 记录最终状态到TensorBoard
    writer.add_text('Training/final_optimizer', final_optimizer_type, global_step)
    writer.add_scalar('Training/final_global_step', global_step, global_step)
    
    writer.close()
    log_file.close()
