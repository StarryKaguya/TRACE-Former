#!/usr/bin/env python3
"""
RQ-VAE训练脚本 - 将多模态embedding转换为semantic id
基于提供的环境变量进行训练，生成semantic id供后续baseline模型使用
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader

from model_rqvae import MmEmbDataset, RQVAE
from config import get_rqvae_config


def train_rqvae_for_feature(feature_id, device='cuda', epochs=30, batch_size=512, lr=1e-3, enable_mixed_precision=False, enable_monitoring=True):
    """
    为指定特征训练RQ-VAE模型
    
    Args:
        feature_id: 特征ID ('81' 或 '82')
        device: 训练设备
        epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率
        enable_mixed_precision: 是否启用混合精度训练
    """
    print(f"\n========== 开始训练特征 {feature_id} 的 RQ-VAE ==========")
    
    # 使用简单的print输出，移除复杂的进度条
    
    # 获取环境变量
    data_dir = os.environ.get('TRAIN_DATA_PATH')
    ckpt_dir = os.environ.get('TRAIN_CKPT_PATH')
    cache_dir = os.environ.get('USER_CACHE_PATH')
    
    if not data_dir or not ckpt_dir:
        raise ValueError("请确保设置了TRAIN_DATA_PATH, TRAIN_CKPT_PATH环境变量")
    
    # 加载数据
    print(f"加载特征 {feature_id} 的多模态数据...")
    dataset = MmEmbDataset(data_dir, feature_id)
    
    # 🚀 多进程优化DataLoader：数据处理已下沉到Dataset，可安全启用多进程
    num_workers = min(4, os.cpu_count() or 1) if device == 'cuda' else 2  # GPU时适度并行，CPU时保守
    print(f"🚀 启用RQ-VAE训练多进程加速: num_workers={num_workers}")
    print("⚡ 数据处理已优化，多进程将提供真实加速")
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=(device == 'cuda'),  # GPU时启用pin_memory
        prefetch_factor=2 if num_workers > 0 else 2,  # 提前预取
        persistent_workers=True if num_workers > 0 else False  # 保持worker进程
    )
    
    # 获取配置
    config = get_rqvae_config()[feature_id]
    print(f"📊 模型配置: 潜在维度={config['latent_dim']}, codebook数量={config['num_codebooks']}, 码本大小={config['codebook_size']}")
    
    # 初始化模型
    print("🎯 初始化改进的RQ-VAE模型...")
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
    
    # 🚀 模型参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 模型参数: 总计={total_params:,}, 可训练={trainable_params:,}")
    
    # 🎯 改进的优化器配置 - 稳定的学习率和调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)  # 恢复正常学习率
    
    # 使用warmup + cosine调度器
    warmup_steps = len(dataloader) * 2  # 前2个epoch进行warmup
    total_steps = len(dataloader) * epochs
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # 🚀 混合精度训练（可选）
    use_amp = enable_mixed_precision and device == 'cuda' and hasattr(torch.cuda, 'amp')
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
        print("✅ 启用混合精度训练，加速GPU计算")
    elif enable_mixed_precision and device != 'cuda':
        print("⚠️ 混合精度训练仅支持CUDA，当前使用CPU模式")
    elif enable_mixed_precision and not hasattr(torch.cuda, 'amp'):
        print("⚠️ 当前PyTorch版本不支持混合精度训练")
    else:
        print("📝 使用标准精度训练")
    
    # 🎯 改进的训练循环 - 添加码本健康监控
    print("🚀 开始改进的RQ-VAE训练...")
    model.train()
    rq_global_step = 0
    
    # 训练监控变量
    best_total_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(epochs):
        epoch_recon_loss = 0.0
        epoch_vq_loss = 0.0
        epoch_total_loss = 0.0
        epoch_commitment_loss = 0.0
        epoch_diversity_loss = 0.0
        epoch_perplexity = 0.0
        
        print(f"🚀 Epoch {epoch+1}/{epochs} 开始训练...")
        
        for batch_idx, (tid_batch, emb_batch) in enumerate(dataloader):
            emb_batch = emb_batch.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # 🚀 混合精度前向传播
            if use_amp:
                with torch.cuda.amp.autocast():
                    x_hat, semantic_id_list, recon_loss, vq_loss, total_loss, stats = model(emb_batch)
                # 混合精度反向传播
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # 标准训练
                x_hat, semantic_id_list, recon_loss, vq_loss, total_loss, stats = model(emb_batch)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            scheduler.step()  # 每个step更新学习率
            rq_global_step += 1
            
            # 记录损失
            epoch_recon_loss += recon_loss.item()
            epoch_vq_loss += vq_loss.item()
            epoch_total_loss += total_loss.item()
            epoch_commitment_loss += stats.get('commitment_loss', 0.0)
            epoch_diversity_loss += stats.get('diversity_loss', 0.0)
            epoch_perplexity += stats.get('avg_perplexity', 0.0)

            # 🎯 详细监控：每10个batch打印一次状态
            if batch_idx % 10 == 0 or batch_idx == len(dataloader) - 1:
                print(f"📊 Batch {batch_idx+1}/{len(dataloader)}: "
                      f"recon={recon_loss.item():.4f}, "
                      f"vq={vq_loss.item():.4f}, "
                      f"total={total_loss.item():.4f}, "
                      f"perplexity={stats.get('avg_perplexity', 0.0):.2f}, "
                      f"usage={stats.get('avg_usage_rate', 0.0):.3f}, "
                      f"lr={optimizer.param_groups[0]['lr']:.1e}")
                
                # 🚨 码本健康检查
                if enable_monitoring and batch_idx % 50 == 0:
                    codebook_stats = model.get_codebook_stats()
                    for cb_name, cb_stats in codebook_stats.items():
                        if cb_stats['usage_rate'] < 0.1:  # 使用率低于10%
                            print(f"⚠️ {cb_name}: 使用率过低 {cb_stats['usage_rate']:.3f}, 死码数量: {cb_stats['dead_codes']}")
                        elif cb_stats['normalized_entropy'] < 0.5:  # 熵过低
                            print(f"⚠️ {cb_name}: 多样性不足 entropy={cb_stats['normalized_entropy']:.3f}")
        
        # 打印epoch统计
        n_batches = len(dataloader)
        avg_recon = epoch_recon_loss / n_batches
        avg_vq = epoch_vq_loss / n_batches
        avg_total = epoch_total_loss / n_batches
        avg_commitment = epoch_commitment_loss / n_batches
        avg_diversity = epoch_diversity_loss / n_batches
        avg_perplexity = epoch_perplexity / n_batches
        
        print(f"✅ Epoch {epoch+1}/{epochs} 完成:")
        print(f"   重建损失: {avg_recon:.4f}, VQ损失: {avg_vq:.4f}, 总损失: {avg_total:.4f}")
        print(f"   承诺损失: {avg_commitment:.4f}, 多样性损失: {avg_diversity:.4f}, 平均困惑度: {avg_perplexity:.2f}")
        
        # 🎯 详细的码本健康报告
        if enable_monitoring:
            print(f"📊 码本健康报告 (Epoch {epoch+1}):")
            codebook_stats = model.get_codebook_stats()
            for cb_name, cb_stats in codebook_stats.items():
                print(f"   {cb_name}: 使用率={cb_stats['usage_rate']:.3f}, "
                      f"熵={cb_stats['normalized_entropy']:.3f}, "
                      f"死码={cb_stats['dead_codes']}, "
                      f"总使用={cb_stats['total_usage']:.0f}")
        
        # Early stopping检查
        if avg_total < best_total_loss:
            best_total_loss = avg_total
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"🛑 Early stopping: 连续{patience}个epoch无改进")
            break
        
        # 🚀 GPU内存清理
        if device == 'cuda':
            torch.cuda.empty_cache()
        
        # 每5个epoch保存一次模型
        if (epoch + 1) % 5 == 0 and cache_dir:
            save_dir = Path(cache_dir) / f"global_step{rq_global_step}.rqvae_feat_{feature_id}_epoch_{epoch+1}"
            save_dir.mkdir(parents=True, exist_ok=True)
            model_path = save_dir / "model.pt"
            torch.save(model.state_dict(), model_path)
            print(f"📁 Epoch {epoch+1} 模型已保存到: {model_path}")
    
    # 保存最终模型
    if cache_dir:  # 添加cache_dir检查
        final_save_dir = Path(cache_dir, f"global_step{rq_global_step}.rqvae_feat_{feature_id}_final")
        final_save_dir.mkdir(parents=True, exist_ok=True)
        final_model_path = final_save_dir / "model.pt"
        torch.save(model.state_dict(), final_model_path)
        print(f"最终模型已保存到: {final_model_path}")
    else:
        print("警告: 未设置cache_dir，模型未保存")
    print(f"✅ 特征 {feature_id} 的 RQ-VAE 训练完成！")
    return


def main():
    parser = argparse.ArgumentParser(description='RQ-VAE训练脚本')
    parser.add_argument('--features', nargs='+', default=['81', '82'], 
                       choices=['81', '82', '83', '84', '85', '86'], help='要训练的特征ID（支持81-86）')
    parser.add_argument('--device', default='cuda', help='训练设备')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数（默认增加到30）')
    parser.add_argument('--batch_size', type=int, default=512, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--enable_mixed_precision', action='store_true', 
                       help='Enable mixed precision training (AMP) for faster GPU training')
    parser.add_argument('--enable_monitoring', action='store_true', default=True,
                       help='启用码本健康监控（默认开启）')
    parser.add_argument('--auto_precompute', action='store_true', default=False,
                       help='训练完成后自动运行预计算semantic_id脚本')
    
    args = parser.parse_args()
    
    print("RQ-VAE训练开始...")
    print(f"训练特征: {args.features}")
    print(f"设备: {args.device}")
    print(f"轮数: {args.epochs}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"混合精度: {args.enable_mixed_precision}")
    print(f"健康监控: {args.enable_monitoring}")
    print(f"自动预计算: {args.auto_precompute}")
    
    # 为每个特征训练改进的RQ-VAE
    for feature_id in args.features:
        print(f"\n🎯 开始训练特征 {feature_id} 的改进RQ-VAE")
        train_rqvae_for_feature(
            feature_id=feature_id,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            enable_mixed_precision=args.enable_mixed_precision,
            enable_monitoring=args.enable_monitoring
        )
        print(f"✅ 特征 {feature_id} 的 RQ-VAE 训练完成")
    
    print("\n🎉 RQ-VAE训练全部完成!")
    
    # 可选：自动运行预计算semantic_id脚本
    if args.auto_precompute:
        print("\n🎯 开始自动预计算semantic_id...")
        try:
            import subprocess
            import sys
            
            cmd = [
                sys.executable, 'precompute_semantic_ids.py',
                '--features'] + args.features + [
                '--device', args.device,
                '--batch_size', str(args.batch_size)
            ]
            
            print(f"🚀 执行命令: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("✅ 预计算semantic_id完成")
            print(result.stdout)
            
        except subprocess.CalledProcessError as e:
            print(f"❌ 预计算semantic_id失败: {e}")
            print(f"错误输出: {e.stderr}")
        except Exception as e:
            print(f"❌ 预计算semantic_id异常: {e}")
    
    print("💡 端到端模式：semantic_id将在BaselineModel运行时实时生成")
    print("🔄 接下来可以运行baseline模型训练，RQ-VAE权重将被自动加载并冻结")
    if args.auto_precompute:
        print("🎯 或者使用 --use_precomputed_semantic_ids 参数启用预计算模式")


if __name__ == "__main__":
    main()
