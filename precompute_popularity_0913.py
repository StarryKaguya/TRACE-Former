#!/usr/bin/env python3
"""
离线流行度预计算程序

功能：
1. 扫描训练数据，统计物品出现次数
2. 生成流行度判断表，支持快速查询
3. 保存预计算结果到cache目录，避免每次训练重复计算

使用方法：
    python precompute_popularity.py --data_dir /path/to/data --min_count 5
"""

import json
import pickle
import argparse
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm


class PopularityPrecomputer:
    """
    流行度预计算器 - 简化版本，只统计计数
    """
    
    def __init__(self, data_dir, popular_ratio=0.15):
        """
        初始化预计算器
        
        Args:
            data_dir: 数据目录路径
            popular_ratio: 流行物品占比（默认0.15，即15%）
        """
        self.data_dir = Path(data_dir)
        self.popular_ratio = popular_ratio
        
        # 加载必要的数据文件
        self._load_metadata()
        
    def _load_metadata(self):
        """加载数据集元信息"""
        print("📊 加载数据集元信息...")
        
        # 加载偏移量文件
        with open(self.data_dir / 'seq_offsets.pkl', 'rb') as f:
            self.seq_offsets = pickle.load(f)
        
        # 加载物品特征字典
        self.item_feat_dict = json.load(open(self.data_dir / "item_feat_dict.json", 'r'))
        
        # 加载索引器
        with open(self.data_dir / 'indexer.pkl', 'rb') as f:
            indexer = pickle.load(f)
            self.itemnum = len(indexer['i'])
            self.usernum = len(indexer['u'])
            # 🎯 重要：保存item_id -> creative_id的映射
            self.indexer_i_rev = {v: k for k, v in indexer['i'].items()}
        
        self.data_file_path = self.data_dir / "seq.jsonl"
        
        print(f"✅ 元信息加载完成: {self.usernum} users, {self.itemnum} items")
    
    def _load_user_data(self, uid, data_file):
        """
        从数据文件中加载单个用户的数据
        
        Args:
            uid: 用户ID(reid)
            data_file: 已打开的数据文件句柄
            
        Returns:
            data: 用户序列数据
        """
        data_file.seek(self.seq_offsets[uid])
        line = data_file.readline()
        data = json.loads(line)
        return data
    
    def compute_item_stats(self):
        """
        计算物品和用户统计信息，包括流行度、CTR和用户活跃度
        
        Returns:
            tuple: (item_counts, item_clicks, item_ctr, user_counts, user_clicks, user_ctr) 
        """
        print("🔍 扫描全量数据，统计物品流行度、CTR和用户特征...")
        
        total_users = len(self.seq_offsets)
        item_counts = {}  # 物品曝光次数
        item_clicks = {}  # 物品点击次数
        user_counts = {}  # 用户曝光次数（用户活跃度）
        user_clicks = {}  # 用户点击次数
        
        with open(self.data_file_path, 'rb') as data_file:
            for uid in tqdm(range(total_users), desc="统计物品和用户特征"):
                try:
                    user_sequence = self._load_user_data(uid, data_file)
                    for record_tuple in user_sequence:
                        # 🔧 修复：数据格式为(user_id, item_id, user_feat, item_feat, action_type, timestamp)
                        user_id, item_id, _, _, action_type, _ = record_tuple
                        
                        if item_id and item_id != 0:
                            # 统计物品曝光（所有action_type都算曝光）
                            item_counts[item_id] = item_counts.get(item_id, 0) + 1
                            
                            # 统计物品点击（action_type=1表示点击）
                            if action_type == 1:
                                item_clicks[item_id] = item_clicks.get(item_id, 0) + 1
                        
                        if user_id and user_id != 0:
                            # 统计用户曝光（用户活跃度）
                            user_counts[user_id] = user_counts.get(user_id, 0) + 1
                            
                            # 统计用户点击
                            if action_type == 1:
                                user_clicks[user_id] = user_clicks.get(user_id, 0) + 1
                                
                except Exception as e:
                    # 跳过可能的损坏数据
                    continue
        
        # 计算原始item CTR（用于后续贝叶斯平滑）
        item_ctr = {}
        for item_id in item_counts:
            # 🎯 转换为creative_id作为键，确保与dataset.py一致
            creative_id = self.indexer_i_rev.get(item_id, None)
            if creative_id:  # 只有有效的creative_id才处理
                clicks = item_clicks.get(item_id, 0)
                exposures = item_counts[item_id]
                item_ctr[creative_id] = clicks / exposures if exposures > 0 else 0.0
        
        # 计算用户CTR
        user_ctr = {}
        for user_id in user_counts:
            clicks = user_clicks.get(user_id, 0)
            exposures = user_counts[user_id]
            user_ctr[user_id] = clicks / exposures if exposures > 0 else 0.0
        
        print(f"✅ 统计完成: 发现 {len(item_counts)} 个活跃物品, {len(user_counts)} 个活跃用户")
        print(f"📊 物品CTR统计: 总点击数 {sum(item_clicks.values())}, 总曝光数 {sum(item_counts.values())}")
        print(f"📊 用户CTR统计: 平均用户活跃度 {np.mean(list(user_counts.values())):.2f}")
        
        return item_counts, item_clicks, item_ctr, user_counts, user_clicks, user_ctr
    
    def build_feature_tables(self, item_counts, item_clicks, item_ctr, user_counts, user_clicks, user_ctr, bayesian_smooth_c=20):
        """
        构建流行度判断表和CTR特征表（包含贝叶斯平滑）
        
        Args:
            item_counts: {item_id: count} 物品计数字典
            item_clicks: {item_id: clicks} 物品点击字典
            item_ctr: {item_id: ctr} 物品原始CTR字典
            user_counts: {user_id: count} 用户计数字典
            user_clicks: {user_id: clicks} 用户点击字典
            user_ctr: {user_id: ctr} 用户CTR字典
            bayesian_smooth_c: 贝叶斯平滑参数C，默认20
            
        Returns:
            dict: 包含流行度和CTR数据的字典
        """
        print(f"🔧 构建流行度判断表，目标流行物品占比: {self.popular_ratio*100:.1f}%...")
        print(f"🎯 使用贝叶斯平滑参数 C={bayesian_smooth_c}")
        
        # 统计信息
        total_items = len(item_counts)
        
        # 计算分位数阈值
        counts = list(item_counts.values())
        counts.sort(reverse=True)  # 降序排列
        
        # 计算目标分位数位置
        target_index = int(total_items * self.popular_ratio)
        if target_index >= total_items:
            target_index = total_items - 1
        
        # 获取分位数阈值
        quantile_threshold = counts[target_index] if counts else 1
        
        # 构建流行物品集合
        popular_items = {item_id: count for item_id, count in item_counts.items() 
                        if count >= quantile_threshold}
        
        actual_ratio = len(popular_items) / total_items if total_items > 0 else 0
        
        # 🎯 计算全局平均CTR作为贝叶斯平滑的先验
        total_clicks = sum(item_clicks.values())
        total_exposures = sum(item_counts.values())
        global_avg_ctr = total_clicks / total_exposures if total_exposures > 0 else 0.05
        
        # 🎯 应用贝叶斯平滑计算item CTR
        smoothed_item_ctr = {}
        for item_id in item_counts:
            creative_id = self.indexer_i_rev.get(item_id, None)
            if creative_id:
                clicks = item_clicks.get(item_id, 0)
                exposures = item_counts[item_id]
                # 贝叶斯平滑公式: (clicks + C * global_avg_ctr) / (exposures + C)
                smoothed_ctr = (clicks + bayesian_smooth_c * global_avg_ctr) / (exposures + bayesian_smooth_c)
                smoothed_item_ctr[creative_id] = smoothed_ctr
        
        # 🎯 计算用户CTR的平滑版本
        user_total_clicks = sum(user_clicks.values())
        user_total_exposures = sum(user_counts.values())
        global_user_avg_ctr = user_total_clicks / user_total_exposures if user_total_exposures > 0 else global_avg_ctr
        
        smoothed_user_ctr = {}
        for user_id in user_counts:
            clicks = user_clicks.get(user_id, 0)
            exposures = user_counts[user_id]
            # 对用户CTR也应用贝叶斯平滑
            smoothed_ctr = (clicks + bayesian_smooth_c * global_user_avg_ctr) / (exposures + bayesian_smooth_c)
            smoothed_user_ctr[user_id] = smoothed_ctr
        
        # 保留原始CTR中位数用于兼容性
        ctr_values = list(item_ctr.values())
        ctr_values = [ctr for ctr in ctr_values if ctr > 0]  # 过滤掉0值
        median_ctr = np.median(ctr_values) if ctr_values else global_avg_ctr
        
        # 🎯 新增：计算基于频率的采样权重 (alpha=0.75)
        creative_id_counts = {}
        for item_id, count in item_counts.items():
            creative_id = self.indexer_i_rev.get(item_id)
            if creative_id:
                creative_id_counts[creative_id] = count

        if creative_id_counts:
            alpha = 0.75
            cids = list(creative_id_counts.keys())
            counts_array = np.array(list(creative_id_counts.values()), dtype=np.float64)
            weights = counts_array ** alpha
            item_sampling_weights = dict(zip(cids, weights))
            print(f"⚖️  加权采样权重计算完成 (alpha={alpha})")
            print(f"   采样权重物品数: {len(item_sampling_weights)}")
        else:
            item_sampling_weights = {}
            print("⚠️  未能计算加权采样权重，物品计数为空")
        
        print(f"📊 流行度统计:")
        print(f"   总物品数: {total_items}")
        print(f"   分位数阈值: {quantile_threshold}")
        print(f"   流行物品数: {len(popular_items)} (阈值>={quantile_threshold})")
        print(f"   实际流行度比例: {actual_ratio*100:.1f}%")
        print(f"📊 CTR统计:")
        print(f"   有效CTR物品数: {len(ctr_values)}")
        print(f"   全局平均CTR: {global_avg_ctr:.4f}")
        print(f"   原始CTR中位数: {median_ctr:.4f}")
        print(f"   平滑后CTR范围: {min(smoothed_item_ctr.values()):.4f} - {max(smoothed_item_ctr.values()):.4f}" if smoothed_item_ctr else "   平滑CTR范围: 无有效数据")
        print(f"📊 用户特征统计:")
        print(f"   活跃用户数: {len(user_counts)}")
        print(f"   全局用户平均CTR: {global_user_avg_ctr:.4f}")
        print(f"   平滑后用户CTR范围: {min(smoothed_user_ctr.values()):.4f} - {max(smoothed_user_ctr.values()):.4f}" if smoothed_user_ctr else "   用户CTR范围: 无有效数据")
        
        return {
            'item_counts': item_counts,
            'popularity_judgment': popular_items,
            'min_count': quantile_threshold,
            'total_items': total_items,
            'item_ctr': item_ctr,  # 🎯 原始CTR数据（保留兼容性）
            'smoothed_item_ctr': smoothed_item_ctr,  # 🎯 新增：贝叶斯平滑后的item CTR
            'avg_ctr': median_ctr,   # 🎯 使用中位数作为平均CTR（兼容性）
            'global_avg_ctr': global_avg_ctr,  # 🎯 新增：全局平均CTR
            'user_counts': user_counts,  # 🎯 新增：用户活跃度
            'user_ctr': user_ctr,  # 🎯 新增：原始用户CTR
            'smoothed_user_ctr': smoothed_user_ctr,  # 🎯 新增：平滑后用户CTR
            'global_user_avg_ctr': global_user_avg_ctr,  # 🎯 新增：全局用户平均CTR
            'bayesian_smooth_c': bayesian_smooth_c,  # 🎯 新增：保存平滑参数
            'item_sampling_weights': item_sampling_weights  # 🎯 新增：加权采样权重
        }
    
    def save_popularity_data(self, popularity_data, output_path=None):
        """
        保存预计算的流行度数据到cache目录
        
        Args:
            popularity_data: 流行度数据字典
            output_path: 输出文件路径（可选）
        """
        if output_path is None:
            # 只保存到cache_dir，如果未设置则报错
            cache_dir = os.environ.get('USER_CACHE_PATH')
            if not cache_dir:
                raise ValueError("USER_CACHE_PATH环境变量未设置，无法保存流行度数据")
            output_dir = Path(cache_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "popularity_data_0913.pkl"
        
        print(f"💾 保存流行度数据到: {output_path}")
        
        with open(output_path, 'wb') as f:
            pickle.dump(popularity_data, f)
        
        # 保存统计信息到文本文件
        stats_path = output_path.with_suffix('.txt')
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write(f"流行度和CTR预计算统计信息\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"数据目录: {self.data_dir}\n")
            f.write(f"用户数量: {self.usernum}\n")
            f.write(f"物品数量: {self.itemnum}\n")
            f.write(f"活跃物品数量: {len(popularity_data['item_counts'])}\n")
            f.write(f"分位数阈值: {popularity_data['min_count']}\n")
            f.write(f"目标流行度比例: {self.popular_ratio*100:.1f}%\n")
            actual_ratio = len(popularity_data['popularity_judgment']) / popularity_data['total_items']
            f.write(f"实际流行度比例: {actual_ratio*100:.1f}%\n")
            f.write(f"流行物品数量: {len(popularity_data['popularity_judgment'])}\n")
            
            # CTR统计信息
            f.write(f"\nCTR统计信息:\n")
            f.write(f"全局平均CTR: {popularity_data.get('global_avg_ctr', 0):.4f}\n")
            f.write(f"贝叶斯平滑参数C: {popularity_data.get('bayesian_smooth_c', 20)}\n")
            f.write(f"原始CTR中位数: {popularity_data['avg_ctr']:.4f}\n")
            f.write(f"有原始CTR的物品数量: {len(popularity_data['item_ctr'])}\n")
            f.write(f"有平滑CTR的物品数量: {len(popularity_data.get('smoothed_item_ctr', {}))}\n")
            if popularity_data.get('smoothed_item_ctr'):
                smoothed_ctr_values = list(popularity_data['smoothed_item_ctr'].values())
                f.write(f"平滑CTR范围: {min(smoothed_ctr_values):.4f} - {max(smoothed_ctr_values):.4f}\n")
            
            # 🎯 新增：用户特征统计信息
            if popularity_data.get('user_counts'):
                f.write(f"\n用户特征统计信息:\n")
                f.write(f"活跃用户数量: {len(popularity_data['user_counts'])}\n")
                f.write(f"全局用户平均CTR: {popularity_data.get('global_user_avg_ctr', 0):.4f}\n")
                f.write(f"有用户CTR的用户数量: {len(popularity_data.get('user_ctr', {}))}\n")
                f.write(f"有平滑用户CTR的用户数量: {len(popularity_data.get('smoothed_user_ctr', {}))}\n")
                if popularity_data.get('smoothed_user_ctr'):
                    user_ctr_values = list(popularity_data['smoothed_user_ctr'].values())
                    f.write(f"平滑用户CTR范围: {min(user_ctr_values):.4f} - {max(user_ctr_values):.4f}\n")
                    f.write(f"平均用户活跃度: {np.mean(list(popularity_data['user_counts'].values())):.2f}\n")
            
            # 🎯 新增：加权采样统计信息
            if popularity_data.get('item_sampling_weights'):
                f.write(f"\n加权采样信息:\n")
                f.write(f"已生成采样权重的物品数: {len(popularity_data['item_sampling_weights'])}\n")
                weights = list(popularity_data['item_sampling_weights'].values())
                f.write(f"权重范围: {min(weights):.4f} - {max(weights):.4f}\n")
            
            # 流行度Top10统计
            if popularity_data['item_counts']:
                f.write(f"\n流行度Top10物品:\n")
                sorted_items = sorted(popularity_data['item_counts'].items(), 
                                    key=lambda x: x[1], reverse=True)[:10]
                for i, (item_id, count) in enumerate(sorted_items, 1):
                    # 🎯 item_id转换为creative_id查找CTR
                    creative_id = self.indexer_i_rev.get(item_id, None)
                    original_ctr = popularity_data['item_ctr'].get(creative_id, 0) if creative_id else 0
                    smoothed_ctr = popularity_data.get('smoothed_item_ctr', {}).get(creative_id, 0) if creative_id else 0
                    f.write(f"{i:2d}. Item {item_id}(creative:{creative_id}): {count} 次交互, 原始CTR: {original_ctr:.4f}, 平滑CTR: {smoothed_ctr:.4f}\n")
        
        print(f"✅ 流行度数据保存完成")
        print(f"📊 统计信息保存到: {stats_path}")
    
    def run(self, output_path=None, bayesian_smooth_c=20):
        """
        执行完整的流行度和CTR预计算流程
        
        Args:
            output_path: 输出文件路径（可选）
            bayesian_smooth_c: 贝叶斯平滑参数C（可选，默认20）
        """
        print("🚀 开始流行度、CTR和用户特征预计算...")
        
        # 1. 统计物品流行度、CTR和用户特征
        item_counts, item_clicks, item_ctr, user_counts, user_clicks, user_ctr = self.compute_item_stats()
        
        # 2. 构建特征表（包含贝叶斯平滑）
        popularity_data = self.build_feature_tables(
            item_counts, item_clicks, item_ctr, 
            user_counts, user_clicks, user_ctr, 
            bayesian_smooth_c
        )
        
        # 3. 保存结果
        self.save_popularity_data(popularity_data, output_path)
        
        print("🎉 流行度、CTR和用户特征预计算完成！")
        return popularity_data


def main():
    parser = argparse.ArgumentParser(description='离线流行度预计算程序')
    parser.add_argument('--data_dir', type=str, 
                       default=os.environ.get('TRAIN_DATA_PATH'),
                       help='数据目录路径 (默认: 从环境变量TRAIN_DATA_PATH获取)')
    parser.add_argument('--popular_ratio', type=float, default=0.15,
                       help='流行物品占比 (默认: 0.15，即15%)')
    parser.add_argument('--output', type=str, default=None,
                       help='输出文件路径 (默认: cache_dir/popularity_data.pkl)')
    
    args = parser.parse_args()
    
    if not args.data_dir:
        print("❌ 错误: 请指定数据目录路径或设置环境变量TRAIN_DATA_PATH")
        return
    
    # 创建预计算器并运行
    precomputer = PopularityPrecomputer(args.data_dir, args.popular_ratio)
    precomputer.run(args.output)


if __name__ == "__main__":
    main()