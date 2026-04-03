import json
import math
import os
import pickle
import struct
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from config import get_time_interval_config
from config import get_semantic_id_config

# 🎯 新增：别名采样器实现
class AliasMethodSampler:
    """
    高效的别名采样器，实现O(1)复杂度的加权随机采样。
    基于Vose's Alias Method算法
    """
    def __init__(self, weights):
        """
        Args:
            weights (dict): {item_id: weight} 的权重字典
        """
        if not weights:
            self.items = np.array([])
            self.prob_table = np.array([])
            self.alias_table = np.array([])
            return

        self.items = np.array(list(weights.keys()))
        probabilities = np.array(list(weights.values()), dtype=np.float64)
        
        # 归一化概率
        probabilities /= probabilities.sum()
        
        n = len(self.items)
        self.prob_table = np.zeros(n)
        self.alias_table = np.zeros(n, dtype=np.int32)

        # Vose's Alias Method 算法
        small = []
        large = []
        for i, p in enumerate(probabilities * n):
            if p < 1.0:
                small.append(i)
            else:
                large.append(i)

        while small and large:
            s = small.pop()
            l = large.pop()

            self.prob_table[s] = probabilities[s] * n
            self.alias_table[s] = l

            probabilities[l] = (probabilities[l] + probabilities[s]) - (1.0 / n)

            if probabilities[l] < 1.0:
                small.append(l)
            else:
                large.append(l)

        for i in large + small:
            self.prob_table[i] = 1.0

    def sample(self):
        """执行一次O(1)采样"""
        if len(self.items) == 0:
            return None
            
        i = np.random.randint(0, len(self.items))
        if np.random.rand() < self.prob_table[i]:
            return self.items[i]
        else:
            return self.items[self.alias_table[i]]

class SemanticIdLoader:
    """
    预计算semantic_id的加载器类
    支持高效的creative_id -> semantic_id映射
    """
    def __init__(self, semantic_id_dict, feature_config):
        """
        🎯 统一语义ID加载器初始化 - 改进版
        Args:
            semantic_id_dict: {creative_id: [semantic_id_list]}
            feature_config: 特征配置字典，支持新的RQ-VAE对齐参数
        """
        self.semantic_id_dict = semantic_id_dict
        self.feature_config = feature_config
        self.array_length = feature_config['array_length']  # 统一使用array_length
        
        # 🎯 使用统一配置中的默认值和padding值
        self.default_value = feature_config.get('default_value', 32)  # 超出有效范围的默认值
        self.padding_value = feature_config.get('padding_value', 33)  # padding值
        
        # 🎯 新增：支持RQ-VAE对齐的配置参数
        self.fusion_mode = feature_config.get('fusion_mode', 'sum')  # sum/concat/weighted_sum/hybrid
        self.enable_layer_weights = feature_config.get('enable_layer_weights', True)
        self.reuse_codebook_weights = feature_config.get('reuse_codebook_weights', True)
        
        # 统计信息
        self.total_items = len(semantic_id_dict)
        print(f"📊 SemanticIdLoader初始化: {self.total_items} items, array_length={self.array_length}")
        print(f"   🎯 融合模式: {self.fusion_mode}, 层权重: {self.enable_layer_weights}, 复用码本: {self.reuse_codebook_weights}")
        print(f"   🔢 default_value={self.default_value}, padding_value={self.padding_value}")
    
    def get_semantic_id(self, creative_id):
        """
        🎯 统一获取指定creative_id的semantic_id
        Args:
            creative_id: 物品ID
        Returns:
            list: semantic_id列表，长度固定为array_length（不足时使用padding_value填充）
        """
        if creative_id in self.semantic_id_dict:
            semantic_ids = self.semantic_id_dict[creative_id].copy()
        else:
            # 🎯 使用统一的默认值作为缺失数据的默认值
            semantic_ids = [self.default_value] * self.array_length
        
        # 确保长度固定为array_length
        if len(semantic_ids) > self.array_length:
            semantic_ids = semantic_ids[:self.array_length]
        elif len(semantic_ids) < self.array_length:
            # 🎯 使用统一的padding_value进行填充
            semantic_ids = semantic_ids + [self.padding_value] * (self.array_length - len(semantic_ids))
        
        return semantic_ids
    
    def get_batch_semantic_ids(self, creative_ids):
        """
        批量获取semantic_id
        Args:
            creative_ids: creative_id列表
        Returns:
            list: semantic_id列表的列表
        """
        return [self.get_semantic_id(cid) for cid in creative_ids]
    
    def validate_semantic_ids(self, sample_size=100):
        """
        🎯 新增：验证semantic_id数据的完整性和格式
        Args:
            sample_size: 抽样验证的数据量
        Returns:
            dict: 验证报告
        """
        if not self.semantic_id_dict:
            return {"status": "empty", "message": "语义ID字典为空"}
        
        # 随机抽样验证
        sample_keys = list(self.semantic_id_dict.keys())[:sample_size]
        valid_count = 0
        invalid_cases = []
        
        for creative_id in sample_keys:
            semantic_ids = self.get_semantic_id(creative_id)
            
            # 检查长度
            if len(semantic_ids) != self.array_length:
                invalid_cases.append(f"长度错误: {creative_id} -> {len(semantic_ids)}")
                continue
                
            # 检查数据类型
            if not all(isinstance(sid, int) for sid in semantic_ids):
                invalid_cases.append(f"类型错误: {creative_id} -> {semantic_ids}")
                continue
                
            valid_count += 1
        
        report = {
            "status": "valid" if len(invalid_cases) == 0 else "invalid",
            "total_samples": len(sample_keys),
            "valid_count": valid_count,
            "invalid_count": len(invalid_cases),
            "fusion_mode": self.fusion_mode,
            "array_length": self.array_length,
            "invalid_cases": invalid_cases[:5]  # 只显示前5个错误案例
        }
        
        print(f"🔍 语义ID验证报告: {report['status']}")
        print(f"   样本: {report['valid_count']}/{report['total_samples']} 有效")
        print(f"   融合模式: {report['fusion_mode']}, 数组长度: {report['array_length']}")
        
        return report


class MyDataset(torch.utils.data.Dataset):
    """
    用户序列数据集

    Args:
        data_dir: 数据文件目录， data_path = os.environ.get('TRAIN_DATA_PATH')
        args: 全局参数

    Attributes:
        data_dir: 数据文件目录
        maxlen: 最大长度
        item_feat_dict: 物品特征字典
        mm_emb_ids: 激活的mm_emb特征ID
        mm_emb_dict: 多模态特征字典
        itemnum: 物品数量
        usernum: 用户数量
        indexer_i_rev: 物品索引字典 (reid -> item_id)
        indexer_u_rev: 用户索引字典 (reid -> user_id)
        indexer: 索引字典
        feature_default_value: 特征缺省值
        feature_types: 特征类型，分为user和item的sparse, array, emb, continual类型
        feat_statistics: 特征统计信息，包括user和item的特征数量
    """

    def __init__(self, data_dir, args):
        """
        初始化数据集
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self._load_data_and_offsets()
        self.maxlen = args.maxlen # 最大长度，指的是用户序列的最大长度
        self.mm_emb_ids = args.mm_emb_id 

        self.item_feat_dict = json.load(open(Path(data_dir, "item_feat_dict.json"), 'r'))
        
        self.enable_rqvae = getattr(args, 'enable_rqvae', False)
        self.use_precomputed_semantic_ids = getattr(args, 'use_precomputed_semantic_ids', False)
        
        if self.enable_rqvae and self.use_precomputed_semantic_ids:
            print("🎯 预计算模式：跳过多模态特征加载，将使用semantic_id")
            self.mm_emb_dict = None
        else:
            # 🔧 重构：支持数据类型参数化，默认为train
            data_type = getattr(self, '_get_mm_emb_data_type', lambda: "train")()
            print(f"📊 加载多模态特征用于端到端模式或传统模式（数据类型：{data_type}）")
            self.mm_emb_dict = load_mm_emb(Path(data_dir, "creative_emb"), self.mm_emb_ids, data_type) 

        with open(self.data_dir / 'indexer.pkl', 'rb') as ff:
            indexer = pickle.load(ff)
            self.itemnum = len(indexer['i'])
            self.usernum = len(indexer['u'])
        self.indexer_i_rev = {v: k for k, v in indexer['i'].items()}
        self.indexer_u_rev = {v: k for k, v in indexer['u'].items()}
        self.indexer = indexer

        self.time_config = get_time_interval_config()
        self.enable_time_features = getattr(args, 'enable_time_features', False)  
        self.disable_time_diff_features = getattr(args, 'disable_time_diff_features', False)  
        self.enable_ctr_feature = getattr(args, 'enable_ctr_feature', True)  # 默认启用CTR特征
        
        # 🔧 重构：统一在_init_feat_info中处理所有特征初始化，出了预处理语义ID
        self.feature_default_value, self.feature_types, self.feat_statistics = self._init_feat_info()
        
        self.semantic_id_loaders = {}
        if self.enable_rqvae and self.use_precomputed_semantic_ids:
            self._load_precomputed_semantic_ids() # 初始化语义ID加载器，以及默认值
        elif self.enable_rqvae:
            print("🎯 启用端到端RQ-VAE模式（semantic特征已在_init_feat_info中初始化）")
        else:
            print("📊 传统模式：使用原始多模态特征")

        # 所有非时间类特征ID（user/item的 sparse/array/continual/semantic）
        non_time_feat_ids = []
        for k, feat_list in self.feature_types.items():
            if k not in ('seq_time_sparse', 'seq_time_continual'):
                non_time_feat_ids.extend(feat_list)

        # 预计算两套集合：包含时间特征/不包含时间特征
        self._all_feat_ids_cand = tuple(non_time_feat_ids)  # 候选不含时间
        if self.enable_time_features:
            time_feat_ids = list(self.feature_types.get('seq_time_sparse', [])) + list(self.feature_types.get('seq_time_continual', []))
        else:
            time_feat_ids = []

        self._all_feat_ids_seq = tuple(non_time_feat_ids + time_feat_ids)  # 序列含时间
        
        # 高效的popularity-aware采样 - 业界最佳实践  
        self.enable_popularity_sampling = getattr(args, 'enable_popularity_sampling', False)  # 默认关闭，可选启用
        
        # 🎯 CTR特征数据
        self.item_ctr = {}  # 物品CTR字典
        self.avg_ctr = 0.0  # 平均CTR（用于冷启动，向后兼容）
        self.global_avg_ctr = 0.0  # 全局平均CTR
        
        # 🎯 新增：用户特征数据
        self.user_ctr = {}  # 用户CTR字典
        self.global_user_avg_ctr = 0.0  # 全局用户平均CTR
        self.user_activity = {}  # 用户活跃度字典
        self.avg_user_activity = 1.0  # 平均用户活跃度
        
        # 🎯 新增：别名采样器相关属性
        self.alias_sampler = None
        self.enable_alias_sampling = getattr(args, 'enable_alias_sampling', False)  # 默认启用别名采样加速
        
        # 🎯 传统加权采样相关属性
        self.weighted_items = None
        self.weighted_probs = None
        
        # 🚀 尝试加载预计算的流行度数据
        if self.enable_popularity_sampling or self.enable_ctr_feature or self.enable_alias_sampling:
            success = self._load_precomputed_popularity()
            if not success:
                print("⚠️ 流行度采样已启用但预计算数据加载失败")
                print("💡 建议：运行 'python precompute_popularity.py' 生成预计算文件")
                print("🔄 将使用均匀随机采样作为fallback")

    def _get_mm_emb_data_type(self):
        """
        重构：返回多模态特征的数据类型，子类可以重写
        """
        return "train"

    def _load_data_and_offsets(self):
        """
        加载用户序列数据和每一行的文件偏移量(预处理好的), 用于快速随机访问数据并I/O
        多进程修复：不再持有全局文件句柄，避免worker进程间的文件指针冲突
        """
        # 🔧 移除全局文件句柄，改为在读取时独立打开
        self.data_file_path = self.data_dir / "seq.jsonl"
        with open(Path(self.data_dir, 'seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)

    def _load_user_data(self, uid):
        """
        从数据文件中加载单个用户的数据
        🔧 多进程修复：每次独立打开文件，避免worker进程间的文件指针冲突

        Args:
            uid: 用户ID(reid)

        Returns:
            data: 用户序列数据，格式为[(user_id, item_id, user_feat, item_feat, action_type, timestamp)]
        """
        # 🔧 多进程安全：每次独立打开文件句柄
        with open(self.data_file_path, 'rb') as data_file:
            data_file.seek(self.seq_offsets[uid])
            line = data_file.readline()
            data = json.loads(line)
        return data

    def _load_precomputed_popularity(self):
        """
        🚀 加载预计算的流行度数据
        如果加载失败，会fallback到实时计算模式
        """
        # 从cache_dir加载预计算数据
        cache_dir = os.environ.get('USER_CACHE_PATH')
        if not cache_dir:
            print("❌ USER_CACHE_PATH环境变量未设置，无法加载预计算流行度数据")
            return False
        precomputed_path = Path(cache_dir) / "popularity_data_0913.pkl"
        
        if not (self.enable_popularity_sampling or self.enable_ctr_feature or self.enable_alias_sampling):
            print("📊 注意：退出load_precomputed_popularity")
            return False
            
        if not precomputed_path.exists():
            print(f"⚠️ 预计算流行度文件不存在: {precomputed_path}")
            print("💡 提示: 运行 'python precompute_popularity.py' 生成预计算文件")
            return False
        
        try:
            print(f"🚀 加载预计算流行度数据: {precomputed_path}")
            with open(precomputed_path, 'rb') as f:
                popularity_data = pickle.load(f)
            
            # 验证数据完整性
            required_keys = ['item_counts', 'min_count', 'total_items']
            if not all(key in popularity_data for key in required_keys):
                print("❌ 预计算数据格式不完整")
                return False
            
            # 🎯 条件性加载CTR数据（仅在启用CTR特征时）
            if self.enable_ctr_feature:
                # 加载物品CTR数据
                if 'smoothed_item_ctr' in popularity_data:
                    # 优先使用平滑后的CTR
                    self.item_ctr = popularity_data['smoothed_item_ctr']
                    self.global_avg_ctr = popularity_data.get('global_avg_ctr', 0.05)
                    print(f"🎯 平滑物品CTR数据加载成功：{len(self.item_ctr)}个物品，全局平均CTR={self.global_avg_ctr:.4f}")
                elif 'item_ctr' in popularity_data:
                    # 回退到原始CTR
                    self.item_ctr = popularity_data['item_ctr']
                    self.global_avg_ctr = popularity_data.get('avg_ctr', 0.05)
                    print(f"🎯 原始物品CTR数据加载成功：{len(self.item_ctr)}个物品，中位数CTR={self.global_avg_ctr:.4f}")
                else:
                    print("⚠️ 预计算数据中未找到物品CTR信息")
                    self.item_ctr = {}
                    self.global_avg_ctr = 0.05
                
                # 🎯 新增：加载用户CTR数据
                if 'smoothed_user_ctr' in popularity_data:
                    # 优先使用平滑后的用户CTR
                    self.user_ctr = popularity_data['smoothed_user_ctr']
                    self.global_user_avg_ctr = popularity_data.get('global_user_avg_ctr', self.global_avg_ctr)
                    print(f"🎯 平滑用户CTR数据加载成功：{len(self.user_ctr)}个用户，全局用户平均CTR={self.global_user_avg_ctr:.4f}")
                elif 'user_ctr' in popularity_data:
                    # 回退到原始用户CTR
                    self.user_ctr = popularity_data['user_ctr']
                    self.global_user_avg_ctr = popularity_data.get('global_user_avg_ctr', self.global_avg_ctr)
                    print(f"🎯 原始用户CTR数据加载成功：{len(self.user_ctr)}个用户")
                else:
                    print("⚠️ 预计算数据中未找到用户CTR信息")
                    self.user_ctr = {}
                    self.global_user_avg_ctr = self.global_avg_ctr
                
                # 🎯 新增：加载用户活跃度数据
                if 'user_counts' in popularity_data:
                    self.user_activity = popularity_data['user_counts']
                    self.avg_user_activity = np.mean(list(self.user_activity.values())) if self.user_activity else 1.0
                    print(f"🎯 用户活跃度数据加载成功：{len(self.user_activity)}个用户，平均活跃度={self.avg_user_activity:.2f}")
                else:
                    print("⚠️ 预计算数据中未找到用户活跃度信息")
                    self.user_activity = {}
                    self.avg_user_activity = 1.0
                
                # 🎯 更新特征默认值
                self.feature_default_value['item_ctr'] = self.global_avg_ctr
                if hasattr(self, 'feature_default_value'):
                    self.feature_default_value['user_ctr'] = self.global_user_avg_ctr
                    self.feature_default_value['user_activity'] = self.avg_user_activity
                
                # 保持向后兼容性
                self.avg_ctr = self.global_avg_ctr
                
            else:
                print("📋 CTR特征已禁用，跳过CTR数据加载")
                self.item_ctr = {}
                self.user_ctr = {}
                self.user_activity = {}
                self.global_avg_ctr = 0.0
                self.global_user_avg_ctr = 0.0
                self.avg_user_activity = 0.0
                self.avg_ctr = 0.0
            
            # 🎯 新增：初始化别名采样器
            if (self.enable_popularity_sampling and 
                self.enable_alias_sampling and 
                'item_sampling_weights' in popularity_data):
                # 使用 creative_id -> reid 的映射转换权重字典的键
                weights_reid = {}
                for creative_id, weight in popularity_data['item_sampling_weights'].items():
                    reid = self.indexer['i'].get(creative_id)
                    if reid:
                        weights_reid[reid] = weight
                
                if weights_reid:
                    self.alias_sampler = AliasMethodSampler(weights_reid)
                    print(f"⚖️  别名采样器初始化完成: {len(self.alias_sampler.items)}个物品")
                else:
                    print("⚠️  未能初始化别名采样器，权重映射为空")
            elif (self.enable_popularity_sampling and 
                  not self.enable_alias_sampling and 
                  'item_sampling_weights' in popularity_data):
                # 🎯 传统加权采样：保存权重数据用于np.random.choice
                weights_reid = {}
                for creative_id, weight in popularity_data['item_sampling_weights'].items():
                    reid = self.indexer['i'].get(creative_id)
                    if reid:
                        weights_reid[reid] = weight
                
                if weights_reid:
                    # 构建用于np.random.choice的数组
                    self.weighted_items = np.array(list(weights_reid.keys()))
                    weights_array = np.array(list(weights_reid.values()))
                    self.weighted_probs = weights_array / weights_array.sum()  # 归一化
                    print(f"⚖️  传统加权采样器初始化完成: {len(self.weighted_items)}个物品")
                else:
                    self.weighted_items = None
                    self.weighted_probs = None
                    print("⚠️  未能初始化传统加权采样器，权重映射为空")
            elif self.enable_popularity_sampling and not self.enable_alias_sampling:
                print("📋 别名采样已禁用，将使用标准加权采样")
            
            print(f"✅ 预计算流行度加载成功:")
            print(f"   - 总物品数: {popularity_data['total_items']}")
            print(f"   - 活跃物品数: {len(popularity_data['item_counts'])}")
            print(f"   - 最小出现次数: {popularity_data['min_count']}")
            if 'item_sampling_weights' in popularity_data:
                print(f"   - 采样权重物品数: {len(popularity_data['item_sampling_weights'])}")
            
            return True
            
        except Exception as e:
            print(f"❌ 加载预计算流行度失败: {e}")
            print("🔄 将fallback到实时计算模式")
            return False

    def _random_neq(self, l, r, s):
        """
        高效的负采样 - 优先使用O(1)的别名采样
        
        Args:
            l: 随机整数的最小值
            r: 随机整数的最大值  
            s: 序列（用户历史交互物品）

        Returns:
            t: 不在序列s中的随机整数
        """
        # 🎯 第一优先级：O(1)别名采样
        if (self.enable_popularity_sampling and 
            self.enable_alias_sampling and 
            self.alias_sampler is not None):
            for _ in range(10):  # 尝试10次，避免死循环
                t = self.alias_sampler.sample()
                if t is not None and t not in s and str(t) in self.item_feat_dict:
                    return t
            print("⚠️ 别名采样多次失败，可能权重分布过于集中")
        
        # 🎯 第二优先级：传统加权采样（O(log N)）
        elif (self.enable_popularity_sampling and 
              not self.enable_alias_sampling and
              self.weighted_items is not None):
            for _ in range(10):  # 尝试10次，避免死循环
                t = np.random.choice(self.weighted_items, p=self.weighted_probs)
                if t not in s and str(t) in self.item_feat_dict:
                    return t
            print("⚠️ 传统加权采样多次失败，可能权重分布过于集中")

        # 🎯 Fallback：均匀随机采样
        for _ in range(10):
            t = np.random.randint(l, r)
            if t not in s and str(t) in self.item_feat_dict:
                return t
            print("⚠️ 均匀随机采样多次失败，可能权重分布过于集中")

        # 最后的保底
        return np.random.randint(l, r)

    def _feat2numpy_internal(self, seq_feature, k):
        """
        Dataset内部工具函数：将seq列表特征转换为numpy array - 从model迁移过来
        专门为单个样本设计，避免batch维度处理
        k: 特征ID
        seq_feature: [maxlen+1]的list，每个元素是特征字典
        """
        # seq_feature是单个样本的特征sequence：[maxlen+1]的list，每个元素是特征字典
        maxlen = len(seq_feature)
        
        # 数组判断条件，包含item_semantic_array
        is_array_feature = (
            k in self.feature_types.get('item_array', []) or 
            k in self.feature_types.get('user_array', []) or
            k in self.feature_types.get('item_semantic_array', [])  
        )
        
        if is_array_feature:
            # Array特征处理 - 找到最大数组长度
            max_array_len = 0
            for item in seq_feature:
                if k in item and item[k] is not None:
                    max_array_len = max(max_array_len, len(item[k]))
            
            if max_array_len == 0:
                max_array_len = 1  # 防止空数组
            
            sample_dtype = np.int64  # 默认int64（适合语义ID）
                
            feat_data = np.zeros((maxlen, max_array_len), dtype=sample_dtype)
            for i, item in enumerate(seq_feature):
                if k in item and item[k] is not None:
                    actual_len = min(len(item[k]), max_array_len)
                    feat_data[i, :actual_len] = item[k][:actual_len]
            
            return feat_data
        else:
            # Sparse/Continual特征处理
            is_continual = (
                k in self.feature_types.get('item_continual', []) or 
                k in self.feature_types.get('user_continual', []) or
                k in self.feature_types.get('seq_time_continual', [])  # 包含绝对时间特征
            )
            feat_data = np.zeros(maxlen, dtype=np.float32 if is_continual else np.int64) # 因为稀疏和数组要过Embedding，所以是整型
            for i, item in enumerate(seq_feature):
                if k in item and item[k] is not None:
                    feat_data[i] = item[k]
            
            return feat_data

    def _process_multimodal_numpy(self, seq_feature, k):
        """
        处理多模态特征的numpy array转换
        """
        # from config import get_rqvae_config
        EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
        
        maxlen = len(seq_feature)
        emb_dim = EMB_SHAPE_DICT.get(k, 32)
        
        feat_data = np.zeros((maxlen, emb_dim), dtype=np.float32)
        for i, item in enumerate(seq_feature):
            if k in item and item[k] is not None:
                if isinstance(item[k], np.ndarray):
                    feat_data[i] = item[k]
                elif isinstance(item[k], list):
                    feat_data[i] = np.array(item[k], dtype=np.float32)
        
        return feat_data

    def __len__(self):
        """
        返回数据集长度，即用户数量

        Returns:
            usernum: 用户数量
        """
        return len(self.seq_offsets)

    def _init_feat_info(self):
        """
        初始化特征信息, 包括特征缺省值和特征类型
        🔧 重构：统一处理所有特征初始化，包括时间特征和semantic特征

        Returns:
            feat_default_value: 特征缺省值，每个元素为字典，key为特征ID，value为特征缺省值
            feat_types: 特征类型，key为特征类型名称，value为包含的特征ID列表
        """

        feat_default_value = {}
        feat_statistics = {}
        feat_types = {}
        feat_types['user_sparse'] = ['103', '104', '105', '109']
        feat_types['item_sparse'] = [ # 默认14个
            '100',
            '117',
            '111',
            '118',
            '101',
            '102',
            '119',
            '120',
            '114',
            '112',
            '121',
            '115',
            '122',
            '116',
        ]
        
        feat_types['item_array'] = []
        feat_types['user_array'] = ['106', '107', '108', '110']
        
        # 条件性注册多模态特征：预计算模式下跳过
        if not (self.enable_rqvae and self.use_precomputed_semantic_ids):
            feat_types['item_emb'] = self.mm_emb_ids
        else:
            feat_types['item_emb'] = []  # 预计算模式不需要多模态特征
            
        # 🎯 条件性添加用户CTR特征：根据开关决定是否启用
        if self.enable_ctr_feature:
            feat_types['user_continual'] = ['user_ctr', 'user_activity']  # 🎯 新增：用户CTR和活跃度特征
            feat_types['item_continual'] = ['item_ctr']  # 🎯 CTR作为item连续特征
        else:
            feat_types['user_continual'] = []
            feat_types['item_continual'] = []  # 不启用CTR特征
        
        if self.enable_time_features:
            time_diff_features = ['time_gap']  # 可以在这里添加更多时间差特征
            time_diff_continuous = ['time_gap_continuous']  # 时间差连续特征
            
            # 绝对时间特征（与时间差无关的绝对时间属性）
            absolute_time_features = [
                # 小时特征（24小时周期）
                'hour_sin', 'hour_cos',
                # 星期特征（7天周期）
                'weekday_sin', 'weekday_cos', 
                # 月份特征（12个月周期）
                'month_sin', 'month_cos',
                # 季节特征（4个季节周期）
                'season_sin', 'season_cos',
                # 年内天数特征（365天周期，捕获年度季节性）
                'day_of_year_sin', 'day_of_year_cos',
                # 工作日/周末标识（二进制特征）
                'is_weekend',
                # 月内天数特征（捕获月度模式，如发薪日效应）
                'day_of_month_sin', 'day_of_month_cos'
            ]
            
            # 根据 disable_time_diff_features 参数决定包含哪些时间特征
            if self.disable_time_diff_features:
                print("🎯 时间差特征隔离：仅使用绝对时间特征，跳过时间差特征")
                # 仅包含绝对时间特征，但保留action_type
                feat_types['seq_time_sparse'] = ['action_type']  # 保留action_type，跳过time_gap
                feat_types['seq_time_continual'] = absolute_time_features
                
                # 🎯 为action_type设置统计信息和默认值（隔离模式下也需要）
                # 映射后：非PAD类别数=2（曝光、点击），模型侧会统一+1创建Embedding
                feat_statistics['action_type'] = 2  # 仅统计非PAD类别，保持与其它稀疏特征一致
                feat_default_value['action_type'] = 0  # PAD值，实际填充时会用1（曝光）
            else:
                # 包含全部时间特征（时间差 + 绝对时间）
                feat_types['seq_time_sparse'] = time_diff_features
                feat_types['seq_time_continual'] = absolute_time_features + time_diff_continuous
                
                # 为时间差特征设置统计信息和默认值
                for feat_id in time_diff_features:
                    if feat_id == 'time_gap':
                        time_gap_buckets = self.time_config['time_gap_buckets']
                        feat_statistics[feat_id] = time_gap_buckets + 1  # +1 为默认值预留空间
                        feat_default_value[feat_id] = time_gap_buckets  # 默认值为最大值，表示"未知/序列开始"
                
                # 🎯 新增：注册 action_type 特征到时间稀疏特征中
                feat_types['seq_time_sparse'].append('action_type')
                # 映射后：非PAD类别数=2（曝光、点击），模型侧会统一+1创建Embedding
                feat_statistics['action_type'] = 2  # 仅统计非PAD类别，保持与其它稀疏特征一致
                feat_default_value['action_type'] = 0  # PAD值，实际填充时会用1（曝光）

            sentinel_value = -2.0
            for feat_id in feat_types['seq_time_continual']:
                feat_default_value[feat_id] = sentinel_value
            print(f"🕐 Dataset时间特征配置完成: seq_time_sparse={list(feat_types.get('seq_time_sparse', []))}, seq_time_continual={list(feat_types.get('seq_time_continual', []))}, 隔离时间差={self.disable_time_diff_features}")
        else:
            feat_types['seq_time_sparse'] = []
            feat_types['seq_time_continual'] = []
        
        # 统一语义ID特征处理：所有RQ-VAE模式都使用数组特征格式
        if self.enable_rqvae:
            # 统一初始化semantic_array特征类型
            feat_types['item_semantic_array'] = []
            semantic_config = get_semantic_id_config()
            active_features = [fid for fid in self.mm_emb_ids if fid in semantic_config['semantic_id_features']]
            
            for feature_id in active_features:
                feature_config = semantic_config['semantic_id_features'][feature_id]
                feature_name = feature_config['feature_name']
                array_length = feature_config['array_length']
                vocab_size = feature_config['vocab_size']
                default_value = feature_config['default_value'] # max_codebook_size
                feat_types['item_semantic_array'].append(feature_name)
                feat_statistics[feature_name] = vocab_size
                feat_default_value[feature_name] = [default_value] * array_length    
                print(f"🎯 注册语义ID特征: {feature_name}, array_length={array_length}, vocab_size={vocab_size}")
        else:
            feat_types['item_semantic_array'] = []

        for feat_id in feat_types['user_sparse']:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id]) 
        for feat_id in feat_types['item_sparse']:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['item_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['user_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['user_continual']:
            if feat_id == 'user_ctr' and self.enable_ctr_feature:
                feat_default_value[feat_id] = 0.05  # 临时默认值，将被预计算的全局用户平均CTR覆盖
            elif feat_id == 'user_activity' and self.enable_ctr_feature:
                feat_default_value[feat_id] = 1.0  # 临时默认值，将被预计算的平均用户活跃度覆盖
            else:
                feat_default_value[feat_id] = 0
        for feat_id in feat_types['item_continual']:
            if feat_id == 'item_ctr' and self.enable_ctr_feature:
                feat_default_value[feat_id] = 0.05  # 临时默认值，将被预计算的中位数覆盖
            else:
                feat_default_value[feat_id] = 0
            
        # 条件性处理多模态特征默认值：预计算模式下跳过
        if feat_types['item_emb']:  
            for feat_id in feat_types['item_emb']:
                SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
                emb_dim = SHAPE_DICT.get(feat_id, 32)
                feat_default_value[feat_id] = np.zeros(emb_dim, dtype=np.float32)
        
        return feat_default_value, feat_types, feat_statistics
    
    def _compute_time_interval_bucket(self, time_delta_seconds):
        """
        将时间间隔（秒）转换为bucket id
        使用对数分桶策略：简单、通用、无需预计算
        
        🚀 优势：
        - 无需预计算和缓存文件
        - 训练测试完全一致
        - 自然处理长尾分布
        - 计算简单高效
        
        Args:
            time_delta_seconds: 时间间隔（秒）
            
        Returns:
            bucket_id: 分桶ID (0 到 time_gap_buckets-1)
        """
        import math
        
        if time_delta_seconds <= 0:
            return 0  # 最小时间间隔对应bucket 0
        
        time_gap_buckets = self.time_config['time_gap_buckets']
        max_time_gap = self.time_config['max_time_gap']
        min_time_gap = self.time_config['min_time_gap']
        
        # 对数分桶：log(time_delta / min_time_gap) / log(max_time_gap / min_time_gap)
        # 映射到 [0, time_gap_buckets-1] 范围
        if time_delta_seconds >= max_time_gap:
            return time_gap_buckets - 1
        
        if time_delta_seconds <= min_time_gap:
            return 0
        
        # 对数映射：将 [min_time_gap, max_time_gap] 映射到 [0, time_gap_buckets-1]
        log_ratio = math.log(time_delta_seconds / min_time_gap) / math.log(max_time_gap / min_time_gap)
        bucket_id = int(log_ratio * (time_gap_buckets - 1))
        
        # 确保在有效范围内
        bucket_id = max(0, min(bucket_id, time_gap_buckets - 1))
        
        return bucket_id

    def _compute_continuous_time_gap(self, time_delta_seconds):
        """
        Args:
            time_delta_seconds: 时间间隔（秒）
            
        Returns:
            continuous_gap: 归一化后的连续时间间隔 [-1.0, 1.0]
                           -1.0 表示无效/缺失时间
        """
        if time_delta_seconds <= 0:
            return -1.0  # 无效时间间隔
        
        # 转换为小时并应用log1p
        time_delta_hours = time_delta_seconds / 3600.0
        log_time = math.log1p(time_delta_hours)  # log(1 + hours)
        
        # 归一化到 [0, 1] 区间，使用配置的上界
        max_norm = self.time_config.get('max_time_gap_norm', 72.0)  # 默认72小时
        max_log = math.log1p(max_norm)
        
        # 归一化并缩放到 [0, 1]
        normalized = min(log_time / max_log, 1.0)
        
        return normalized

    def _compute_absolute_time_features(self, unix_timestamp):
        """
        从Unix时间戳计算扩展的绝对时间特征（sin/cos编码 + 二进制特征）
        🚀 基于业界最佳实践，提供多层次时间模式捕获能力
        🔧 注意：特征类型注册已合并到_init_feat_info中，此方法仅用于运行时计算
        Args:
            unix_timestamp: Unix时间戳（秒）
        Returns:
            tuple: 包含13个时间特征的元组
                  (hour_sin, hour_cos, weekday_sin, weekday_cos, 
                   month_sin, month_cos, season_sin, season_cos,
                   day_of_year_sin, day_of_year_cos, is_weekend,
                   day_of_month_sin, day_of_month_cos)
        """
        import datetime
        import math
        
        # 转换为datetime对象（UTC时间）
        dt = datetime.datetime.fromtimestamp(unix_timestamp, tz=datetime.timezone.utc)
        
        # 1. 小时特征（0-23小时，24小时周期）
        hour = dt.hour
        hour_sin = math.sin(2 * math.pi * hour / 24)
        hour_cos = math.cos(2 * math.pi * hour / 24)
        
        # 2. 星期特征（0-6星期，7天周期）
        weekday = dt.weekday()  # 0=Monday, 6=Sunday
        weekday_sin = math.sin(2 * math.pi * weekday / 7)
        weekday_cos = math.cos(2 * math.pi * weekday / 7)
        
        # 3. 月份特征（1-12月，12个月周期）
        month = dt.month
        month_sin = math.sin(2 * math.pi * (month - 1) / 12)  # 转换为0-11范围
        month_cos = math.cos(2 * math.pi * (month - 1) / 12)
        
        # 4. 季节特征（基于月份计算，4个季节周期）
        # 春季(3-5月)=0, 夏季(6-8月)=1, 秋季(9-11月)=2, 冬季(12-2月)=3
        season = ((month - 3) % 12) // 3
        season_sin = math.sin(2 * math.pi * season / 4)
        season_cos = math.cos(2 * math.pi * season / 4)
        
        # 5. 年内天数特征（1-365/366天，年度周期）
        day_of_year = dt.timetuple().tm_yday
        # 使用365作为标准化因子（闰年的366天会稍微超出周期，但影响很小）
        day_of_year_sin = math.sin(2 * math.pi * (day_of_year - 1) / 365)
        day_of_year_cos = math.cos(2 * math.pi * (day_of_year - 1) / 365)
        
        # 6. 工作日/周末标识（二进制特征）
        is_weekend = float(weekday >= 5)  # 周六(5)和周日(6)为周末
        
        # 7. 月内天数特征（1-31天，月度周期）
        day_of_month = dt.day
        # 使用31作为标准化因子（较短月份会在周期内，保持一致性）
        day_of_month_sin = math.sin(2 * math.pi * (day_of_month - 1) / 31)
        day_of_month_cos = math.cos(2 * math.pi * (day_of_month - 1) / 31)
        
        return (hour_sin, hour_cos, weekday_sin, weekday_cos,
                month_sin, month_cos, season_sin, season_cos,
                day_of_year_sin, day_of_year_cos, is_weekend,
                day_of_month_sin, day_of_month_cos)
    
    def _load_precomputed_semantic_ids(self):
        """
        加载预计算的semantic_id字典
        根据当前数据集类型（训练/推理）自动选择对应的缓存文件
        """

        semantic_config = get_semantic_id_config()
        cache_dir = os.environ.get('USER_CACHE_PATH')
        if not cache_dir:
            raise ValueError("USER_CACHE_PATH环境变量未设置，无法加载semantic_id数据")
        
        # 🔧 重构：使用统一的数据类型获取方法
        data_type = self._get_mm_emb_data_type()
        
        print(f"🎯 加载预计算semantic_id: {data_type}模式")
        
        # 加载每个激活特征的semantic_id
        active_features = [fid for fid in self.mm_emb_ids if fid in semantic_config['semantic_id_features']]
        
        for feature_id in active_features:
            # 🔧 修改：优先使用全覆盖版本（_full后缀）
            cache_file_full = Path(cache_dir) / f"semantic_id_{feature_id}_{data_type}_full.pkl"
            cache_file_regular = Path(cache_dir) / f"semantic_id_{feature_id}_{data_type}.pkl"
            
            cache_file = None
            version_info = ""
            
            # 优先级1：尝试全覆盖版本
            if cache_file_full.exists():
                cache_file = cache_file_full
                version_info = "全覆盖版本"
                print(f"✅ 使用全覆盖版本缓存: {cache_file}")
            # 优先级2：尝试常规版本
            elif cache_file_regular.exists():
                cache_file = cache_file_regular
                version_info = "常规版本"
                print(f"⚠️  使用常规版本缓存（建议升级到全覆盖版本）: {cache_file}")
                print(f"   升级命令: python precompute_semantic_ids.py --features {feature_id} --data_type {data_type} --use_full_coverage")
            # 优先级3：兼容旧的命名规则
            else:
                old_cache_file = Path(cache_dir) / semantic_config['cache_file_pattern'].format(feature_id=feature_id)
                if old_cache_file.exists():
                    cache_file = old_cache_file
                    version_info = "旧格式版本"
                    print(f"⚠️  使用旧格式缓存文件: {cache_file}")
                    print(f"   建议重新运行: python precompute_semantic_ids.py --features {feature_id} --data_type {data_type} --use_full_coverage")
                else:
                    # 生成建议命令
                    suggested_command = f"python precompute_semantic_ids.py --features {feature_id} --data_type {data_type} --use_full_coverage"
                    raise FileNotFoundError(
                        f"预计算semantic_id文件不存在: {cache_file_full}\n"
                        f"也不存在常规版本: {cache_file_regular}\n"
                        f"请先运行: {suggested_command}"
                    )
            
            print(f"📂 加载semantic_id缓存 ({data_type}, {version_info}): {cache_file}")

            with open(cache_file, 'rb') as f:
                semantic_id_dict = pickle.load(f)
            
            # 创建semantic_id加载器
            self.semantic_id_loaders[feature_id] = SemanticIdLoader(
                semantic_id_dict=semantic_id_dict,
                feature_config=semantic_config['semantic_id_features'][feature_id]
            )
            print(f"✅ semantic_id加载完成: 特征{feature_id} ({data_type}, {version_info}), {len(semantic_id_dict)} items")
            
            # 🎯 新增：验证数据完整性
            validation_report = self.semantic_id_loaders[feature_id].validate_semantic_ids()
            if validation_report['status'] != 'valid':
                print(f"⚠️  语义ID数据验证发现问题: {validation_report['invalid_count']} 个无效样本")
                if validation_report['invalid_cases']:
                    print(f"   示例错误: {validation_report['invalid_cases'][:2]}")
            else:
                print(f"✅ 数据验证通过: 融合模式={validation_report['fusion_mode']}")
        

    def _batch_load_multimodal_features(self, item_ids):
        """
        条件性批量预加载多模态特征
        预计算模式下跳过，避免不必要的I/O
        """
        if self.enable_rqvae and self.use_precomputed_semantic_ids:
            return {}  # 预计算模式不需要多模态特征
        
        if not self.mm_emb_dict:
            return {}  # 没有多模态数据
            
        mm_emb_cache = {}
        for item_id in item_ids:
            if item_id == 0:
                continue
            try:
                creative_id = self.indexer_i_rev[item_id]
                mm_emb_cache[item_id] = {}
                for feat_id in self.feature_types.get('item_emb', []):
                    # 🔧 关键修复：creative_id类型转换问题（与infer.py保持一致）
                    emb_vector = self.mm_emb_dict.get(feat_id, creative_id)
                    # 如果失败且creative_id是字符串，尝试转换为整型
                    if emb_vector is None and isinstance(creative_id, str) and creative_id.isdigit():
                        emb_vector = self.mm_emb_dict.get(feat_id, int(creative_id))
                    # 如果失败且creative_id是整型，尝试转换为字符串
                    elif emb_vector is None and isinstance(creative_id, int):
                        emb_vector = self.mm_emb_dict.get(feat_id, str(creative_id))
                    
                    if emb_vector is not None:
                        mm_emb_cache[item_id][feat_id] = emb_vector
            except (KeyError, IndexError):
                continue
        return mm_emb_cache
    
    def fill_missing_feat_cached(self, feat, item_id, mm_emb_cache, include_time_features=True, user_id=None):
        """
        🚀 使用预加载缓存的高效特征填充版本
        
        Args:
            feat: 特征字典
            item_id: 物品ID
            mm_emb_cache: 预加载的多模态特征缓存
            include_time_features: 是否包含时间特征
                - True: 用于序列特征填充（训练/测试的seq）
                - False: 用于候选item特征填充（训练的pos/neg，推理的candidates）
            
        Returns:
            filled_feat: 填充后的特征字典
            
        注意：
            当include_time_features=False时，时间相关特征不会被填充，从而避免数据泄露
            时间相关特征包括：time_gap, hour_sin, hour_cos, weekday_sin, weekday_cos
        """
        if feat == None:
            raise ValueError("Feature dict is None")
        filled_feat = {}
        for k in feat.keys():
            filled_feat[k] = feat[k]

        # 🔧 使用预计算的特征ID集合，避免每次循环构建列表
        all_feat_ids = self._all_feat_ids_seq if include_time_features else self._all_feat_ids_cand
        missing_fields = set(all_feat_ids) - set(feat.keys())

        for feat_id in missing_fields:
            filled_feat[feat_id] = self.feature_default_value[feat_id]
            
        # 🚀 条件性从缓存中获取多模态特征，非预计算模式才需要多模态特征
        if not (self.enable_rqvae and self.use_precomputed_semantic_ids):
            if item_id != 0 and item_id in mm_emb_cache:
                for feat_id in self.feature_types.get('item_emb', []):
                    if feat_id in mm_emb_cache[item_id]:
                        filled_feat[feat_id] = mm_emb_cache[item_id][feat_id]
        
        # 🎯 预计算模式：加载semantic_id（包括冷启动item_id==0，与推理端保持一致）
        if self.enable_rqvae and self.use_precomputed_semantic_ids:
            creative_id = self.indexer_i_rev.get(item_id, None) if item_id != 0 else feat.get('creative_id', 0)
            if creative_id:
                for feat_id, loader in self.semantic_id_loaders.items():
                    semantic_ids = loader.get_semantic_id(creative_id)
                    feature_name = loader.feature_config['feature_name']
                    if semantic_ids is not None:
                        filled_feat[feature_name] = semantic_ids
                        # 🔧 调试：显示冷启动item成功加载semantic_id的情况（与推理端一致）
                        if item_id == 0 and not hasattr(self, '_cold_start_sid_log_count'):
                            self._cold_start_sid_log_count = 0
                        if item_id == 0 and self._cold_start_sid_log_count < 5:
                            print(f"✅ 训练端冷启动item {creative_id} 成功加载semantic_id: {semantic_ids[:3]}...")
                            self._cold_start_sid_log_count += 1
        
        # 🎯 CTR特征填充：仅在启用CTR特征时处理
        if self.enable_ctr_feature:
            # 物品CTR特征填充
            if 'item_ctr' in missing_fields and item_id != 0:
                creative_id = self.indexer_i_rev.get(item_id, None)
                if creative_id and creative_id in self.item_ctr:
                    # 使用预计算的CTR值
                    filled_feat['item_ctr'] = self.item_ctr[creative_id]
                else:
                    # 使用全局平均CTR作为冷启动默认值
                    filled_feat['item_ctr'] = self.global_avg_ctr
            
            # 🎯 新增：用户CTR特征填充
            if 'user_ctr' in missing_fields and user_id is not None and user_id != 0:
                if user_id in self.user_ctr:
                    # 使用预计算的用户CTR值
                    filled_feat['user_ctr'] = self.user_ctr[user_id]
                else:
                    # 使用全局用户平均CTR作为默认值
                    filled_feat['user_ctr'] = self.global_user_avg_ctr
            
            # 🎯 新增：用户活跃度特征填充
            if 'user_activity' in missing_fields and user_id is not None and user_id != 0:
                if user_id in self.user_activity:
                    # 使用预计算的用户活跃度值
                    filled_feat['user_activity'] = float(self.user_activity[user_id])
                else:
                    # 使用平均用户活跃度作为默认值
                    filled_feat['user_activity'] = self.avg_user_activity
        
        return filled_feat


    def __getitem__(self, uid):
        """
        获取单个用户的数据，并进行padding处理，生成模型需要的数据格式
        多进程优化：完成所有CPU数据处理，返回已数值化的tensor字典

        Args:
            uid: 用户ID(reid)
            action_type，1表示item token，2表示user token
        Returns:
            完整的tensor字典，包含所有处理好的特征数据
        """
        user_sequence = self._load_user_data(uid) 

        # 提取时间戳并计算时间间隔（一次循环完成）
        timestamps = []
        time_intervals = []
        last_item_timestamp = None  # 记录上一个item的时间戳，用于计算时间间隔
        ts = set()  # 用户历史交互物品集合，用于负采样时避免重复
        
        ext_user_sequence = []
        for record_tuple in user_sequence: 
            u, i, user_feat, item_feat, action_type, timestamp = record_tuple
            timestamps.append(timestamp if timestamp is not None else 0)
            
            # 处理用户特征
            if u and user_feat:
                ext_user_sequence.insert(0, (u, user_feat, 2, action_type, timestamp))
            
            # 处理物品特征并计算时间间隔
            if i and item_feat:
                ext_user_sequence.append((i, item_feat, 1, action_type, timestamp))
                
                # 同时收集用户交互过的物品ID，用于负采样
                if i != 0:
                    ts.add(i)
                
                # 🕐 在启用时间特征且未隔离时间差特征时，为item交互计算时间间隔
                if self.enable_time_features and not self.disable_time_diff_features:
                    current_timestamp = timestamp if (timestamp is not None and timestamp > 0) else None
                    if last_item_timestamp is None:
                        # 第一个item：无前序上下文，使用默认值表示"序列开始/未知"
                        time_intervals.append((self.time_config['time_gap_buckets'], -1.0))  # (bucket_id, continuous_gap)
                    else:
                        if current_timestamp is not None and last_item_timestamp is not None:
                            # 计算与上一个item的时间间隔
                            time_delta = current_timestamp - last_item_timestamp
                            bucket_id = self._compute_time_interval_bucket(time_delta)
                            continuous_gap = self._compute_continuous_time_gap(time_delta)
                            time_intervals.append((bucket_id, continuous_gap))
                        else:
                            # 时间戳缺失，记作未知桶
                            time_intervals.append((self.time_config['time_gap_buckets'], -1.0))
                    # 更新最后一个item的有效时间戳（仅当本次有效时更新）
                    last_item_timestamp = current_timestamp if current_timestamp is not None else last_item_timestamp

        seq = np.zeros([self.maxlen + 1], dtype=np.int32) 
        pos = np.zeros([self.maxlen + 1], dtype=np.int32)
        neg = np.zeros([self.maxlen + 1], dtype=np.int32)
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        next_token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        next_action_type = np.zeros([self.maxlen + 1], dtype=np.int32) 
        seq_action_type = np.zeros([self.maxlen + 1], dtype=np.int32)  # 🎯 新增：历史token的动作类型

        seq_feat = np.empty([self.maxlen + 1], dtype=object)
        pos_feat = np.empty([self.maxlen + 1], dtype=object)
        neg_feat = np.empty([self.maxlen + 1], dtype=object)
        
        seq_timestamps = np.zeros([self.maxlen + 1], dtype=np.int64)  # 时间戳
        nxt = ext_user_sequence[-1] 
        
        # 第一阶段：收集所有需要的item_id，避免N+1查询；仅限于不是预加载语义ID的情况
        neg_ids = []  
        all_needed_mm_items = set()
        nxt = ext_user_sequence[-1]
        for record_tuple in reversed(ext_user_sequence[:-1]):
            i, feat, type_, act_type, timestamp = record_tuple
            next_i, next_feat, next_type, next_act_type, next_timestamp = nxt
            if i != 0 and type_ == 1:
                all_needed_mm_items.add(i)
            # 预生成负样本ID
            if next_type == 1 and next_i != 0:
                neg_id = self._random_neq(1, self.itemnum + 1, ts)
                neg_ids.append(neg_id)
                all_needed_mm_items.add(neg_id)
            else:
                neg_ids.append(0) # 负样本ID为0，表示没有生成

            nxt = record_tuple  # 正确更新nxt
            if len(neg_ids) >= self.maxlen:  # 限制长度
                break

        # 第二阶段：批量预加载多模态特征到内存缓存
        mm_emb_cache = {}
        mm_emb_cache = self._batch_load_multimodal_features(all_needed_mm_items)
        
        # 第三阶段：填充序列数据
        neg_idx = 0
        idx = self.maxlen 
        time_interval_idx = len(time_intervals) - 2 if len(time_intervals) >= 2 else -1  # 从倒数第二个开始，因为循环跳过最后一个元素
        nxt = ext_user_sequence[-1]  # 初始化nxt为最后一个元素
        
        for record_tuple in reversed(ext_user_sequence[:-1]):
            i, feat, type_, act_type, timestamp = record_tuple 
            next_i, next_feat, next_type, next_act_type, next_timestamp = nxt

            # 🎯 修正：当type_=2时，i才是user_id，而不是uid
            current_user_id = i if type_ == 2 else None  # 只有用户token才需要用户特征
            feat = self.fill_missing_feat_cached(feat, i, mm_emb_cache, include_time_features=True, user_id=current_user_id) 
            # 这里的feat 是一个dict
            if self.enable_time_features and timestamp is not None: 
                # 1. 为item类型添加时间间隔特征（仅在未隔离时间差特征时，user类型不需要时间间隔）
                if (type_ == 1 and not self.disable_time_diff_features and 
                    time_interval_idx >= 0 and time_interval_idx < len(time_intervals)):
                    bucket_id, continuous_gap = time_intervals[time_interval_idx]
                    feat['time_gap'] = bucket_id
                    if self.time_config.get('enable_continuous_time_gap', True):
                        feat['time_gap_continuous'] = continuous_gap
                    time_interval_idx -= 1
                elif type_ == 1 and self.disable_time_diff_features:
                    # 时间差特征隔离模式：不添加time_gap特征，但更新索引
                    if time_interval_idx >= 0:
                        time_interval_idx -= 1 
                
                # 2. 添加扩展的绝对时间特征（所有token类型都需要）
                (hour_sin, hour_cos, weekday_sin, weekday_cos,
                 month_sin, month_cos, season_sin, season_cos,
                 day_of_year_sin, day_of_year_cos, is_weekend,
                 day_of_month_sin, day_of_month_cos) = self._compute_absolute_time_features(timestamp)
                
                # 分配到特征字典
                feat['hour_sin'] = hour_sin
                feat['hour_cos'] = hour_cos
                feat['weekday_sin'] = weekday_sin
                feat['weekday_cos'] = weekday_cos
                feat['month_sin'] = month_sin
                feat['month_cos'] = month_cos
                feat['season_sin'] = season_sin
                feat['season_cos'] = season_cos
                feat['day_of_year_sin'] = day_of_year_sin
                feat['day_of_year_cos'] = day_of_year_cos
                feat['is_weekend'] = is_weekend
                feat['day_of_month_sin'] = day_of_month_sin
                feat['day_of_month_cos'] = day_of_month_cos
            
            if timestamp is not None:
                seq_timestamps[idx] = timestamp
            
            # 更新序列数据
            seq[idx] = i 
            token_type[idx] = type_ 
            next_token_type[idx] = next_type 
            if next_act_type is not None: 
                next_action_type[idx] = next_act_type
            # 🎯 修复：action_type做+1映射避免与padding_idx=0冲突
            # 原始：0=曝光，1=点击；映射后：1=曝光，2=点击，0=PAD
            if act_type is not None:
                seq_action_type[idx] = act_type + 1  # 曝光0→1，点击1→2
            else:
                # 缺失action_type视为曝光（按用户需求）
                seq_action_type[idx] = 1  # 默认为曝光
            seq_feat[idx] = feat
            
            if next_type == 1 and next_i != 0: 
                pos[idx] = next_i
                # 🔄 正样本不包含时间特征：避免数据泄露，确保与推理时一致
                pos_feat_clean = self.fill_missing_feat_cached(next_feat, next_i, mm_emb_cache, include_time_features=False, user_id=None)
                pos_feat[idx] = pos_feat_clean
                
                neg_id = neg_ids[neg_idx] if neg_idx < len(neg_ids) else 0
                neg[idx] = neg_id
                if neg_id != 0:
                    neg_feat_dict = self.fill_missing_feat_cached(self.item_feat_dict.get(str(neg_id)), neg_id, mm_emb_cache, include_time_features=False, user_id=None)
                    # 🔄 负样本不包含时间特征：避免数据泄露
                    neg_feat[idx] = neg_feat_dict
                neg_idx += 1
            nxt = record_tuple  # 更新nxt为当前record，用于下一次迭代
            idx -= 1
            if idx == -1:
                break

        seq_feat = np.where(seq_feat == None, self.feature_default_value, seq_feat) # 将未填到的 object 槽位用 feature_default_value 补齐。
        pos_feat = np.where(pos_feat == None, self.feature_default_value, pos_feat)
        neg_feat = np.where(neg_feat == None, self.feature_default_value, neg_feat)

        # 转换为list以便处理
        seq_feat_list = [seq_feat[i] for i in range(len(seq_feat))]
        pos_feat_list = [pos_feat[i] for i in range(len(pos_feat))]
        neg_feat_list = [neg_feat[i] for i in range(len(neg_feat))]
        
        # ⚡ 关键优化：在Dataset中完成所有特征张量化，避免model中的for循环
        feature_tensors = {}
        
        # 统一处理：与训练集保持完全一致的特征类型处理；思考：没有多模态特征？如果在RQVAE实时加载或者不用RQVAE
        all_feat_types = [
            ('item_sparse', self.feature_types.get('item_sparse', [])),
            ('item_array', self.feature_types.get('item_array', [])),
            ('item_continual', self.feature_types.get('item_continual', [])),
            ('user_sparse', self.feature_types.get('user_sparse', [])),
            ('user_array', self.feature_types.get('user_array', [])),
            ('user_continual', self.feature_types.get('user_continual', [])),
            ('seq_time_sparse', self.feature_types.get('seq_time_sparse', [])),
            ('seq_time_continual', self.feature_types.get('seq_time_continual', [])),  
            ('item_semantic_array', self.feature_types.get('item_semantic_array', [])),
        ]
        
        # 🕐 时间相关特征列表 - 动态构建根据隔离开关
        time_related_features = {
            'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos',
            'month_sin', 'month_cos', 'season_sin', 'season_cos',
            'day_of_year_sin', 'day_of_year_cos', 'is_weekend',
            'day_of_month_sin', 'day_of_month_cos',
        }

        # 时间差特征：仅在未隔离时包含
        if not self.disable_time_diff_features:
            time_related_features.add('time_gap')
            if self.time_config.get('enable_continuous_time_gap', True):
                time_related_features.add('time_gap_continuous')
        
        for feat_type, feat_ids in all_feat_types:
            for feat_id in feat_ids:
                # 🔧 统一逻辑：时间特征只添加到seq中
                if feat_id in time_related_features:
                    # 时间特征：只处理seq
                    feature_tensors[f'seq_{feat_id}'] = self._feat2numpy_internal(seq_feat_list, feat_id)
                else:
                    # 🎯 非时间特征：处理seq、pos、neg（包括semantic特征）
                    feature_tensors[f'seq_{feat_id}'] = self._feat2numpy_internal(seq_feat_list, feat_id)    
                    feature_tensors[f'pos_{feat_id}'] = self._feat2numpy_internal(pos_feat_list, feat_id)
                    feature_tensors[f'neg_{feat_id}'] = self._feat2numpy_internal(neg_feat_list, feat_id)

        if not (self.enable_rqvae and self.use_precomputed_semantic_ids):
            # 端到端RQ-VAE模式或传统模式：处理原始多模态特征
            for feat_id in self.feature_types.get('item_emb', []):
                feature_tensors[f'seq_{feat_id}'] = self._process_multimodal_numpy(seq_feat_list, feat_id)
                feature_tensors[f'pos_{feat_id}'] = self._process_multimodal_numpy(pos_feat_list, feat_id)
                feature_tensors[f'neg_{feat_id}'] = self._process_multimodal_numpy(neg_feat_list, feat_id)

        feature_tensors['seq_timestamp'] = seq_timestamps
        feature_tensors['seq_action_type'] = seq_action_type
        return {
            'seq': seq,
            'pos': pos,
            'neg': neg,
            'token_type': token_type,
            'next_token_type': next_token_type,
            'next_action_type': next_action_type,
            'feature_tensors': feature_tensors
        }   

    @staticmethod
    def collate_fn(batch):
        """
        ⚡ 多进程优化版本：将所有处理好的tensor组装成batch
        Args:
            batch: 多个__getitem__返回的数据字典

        Returns:
            完整的batch tensor字典，直接供model使用
        """
        # 收集基础张量
        seq_list = [item['seq'] for item in batch]
        pos_list = [item['pos'] for item in batch]
        neg_list = [item['neg'] for item in batch]
        token_type_list = [item['token_type'] for item in batch]
        next_token_type_list = [item['next_token_type'] for item in batch]
        next_action_type_list = [item['next_action_type'] for item in batch]
        
        # 转换为batch tensor
        batch_data = {
            'seq': torch.from_numpy(np.array(seq_list)),
            'pos': torch.from_numpy(np.array(pos_list)),
            'neg': torch.from_numpy(np.array(neg_list)),
            'token_type': torch.from_numpy(np.array(token_type_list)),
            'next_token_type': torch.from_numpy(np.array(next_token_type_list)),
            'next_action_type': torch.from_numpy(np.array(next_action_type_list))
        }
        
        # 收集特征张量 - 按特征ID分组，动态补齐array特征
        feature_tensors = {}
        if batch:  # 确保batch不为空
            # 获取所有特征键
            feature_keys = set()
            for item in batch:
                feature_keys.update(item['feature_tensors'].keys())
            
            # 为每个特征键构建batch tensor
            for feat_key in feature_keys:
                feat_batch = []
                for item in batch:
                    if feat_key in item['feature_tensors']:
                        feat_batch.append(item['feature_tensors'][feat_key])
                    else:
                        # 如果某个样本缺失该特征，用零填充
                        sample_feat = item['feature_tensors'][list(item['feature_tensors'].keys())[0]]
                        if len(sample_feat.shape) == 1:
                            feat_batch.append(np.zeros_like(sample_feat))
                        else:
                            feat_batch.append(np.zeros_like(sample_feat))
                
                if feat_batch:
                    # 检查是否为array特征（3维：batch, seq_len, array_len）
                    first_tensor = feat_batch[0]
                    if first_tensor.ndim == 2 and first_tensor.shape[0] > 1:  # 可能是array特征
                        # 找到最大array长度
                        max_array_len = max(tensor.shape[-1] for tensor in feat_batch)
                        # 动态补齐
                        padded_batch = []
                        for tensor in feat_batch:
                            if tensor.shape[-1] < max_array_len:
                                pad_width = ((0, 0), (0, max_array_len - tensor.shape[-1]))
                                padded = np.pad(tensor, pad_width, mode='constant', constant_values=0)
                                padded_batch.append(padded)
                            else:
                                padded_batch.append(tensor)
                        feature_tensors[feat_key] = torch.from_numpy(np.stack(padded_batch))
                    else:
                        # 普通特征直接堆叠
                        feature_tensors[feat_key] = torch.from_numpy(np.array(feat_batch))
        
        batch_data['feature_tensors'] = feature_tensors
        return batch_data


class MyTestDataset(MyDataset):
    """
    测试数据集
    返回 seq, token_type, seq_feat, user_id（无正负样本与“下一步”信息）
    不做采样，只做左填充对齐
    专门处理冷启动（字符串特征→0；新物品→id 0；字符串用户 id 单独返回） 
    """

    def __init__(self, data_dir, args):
        # 🎯 重构：直接调用父类初始化，只需要重写数据类型相关方法
        super().__init__(data_dir, args)
        print(f"✅ MyTestDataset初始化完成：eval模式")

    def _get_mm_emb_data_type(self):
        return "eval"

    def _load_data_and_offsets(self):
        self.data_file_path = self.data_dir / "predict_seq.jsonl"
        with open(Path(self.data_dir, 'predict_seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)

    def _process_cold_start_feat(self, feat):
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

    def __getitem__(self, uid):
        """
        获取单个用户的数据，并进行padding处理，生成模型需要的数据格式
        ⚡ 多进程优化：测试数据集也使用tensor字典格式
        测试集没有点击/曝光特征

        Args:
            uid: 用户在self.data_file中储存的行号
        Returns:
            测试数据的tensor字典
        """
        user_sequence = self._load_user_data(uid)  # 动态加载用户数据，每个元素为 (user_id, item_id, user_feat, item_feat, action_type, timestamp)

        # 🕐 提取时间戳并计算时间间隔（与训练时保持一致，一次循环完成）
        timestamps = []
        time_intervals = []
        last_item_timestamp = None  # 记录上一个item的时间戳，用于计算时间间隔
        
        ext_user_sequence = [] # 代表用户序列，每个元素为 (id, feat, type, action_type, timestamp) - 与训练集保持一致
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, action_type, timestamp = record_tuple
            timestamps.append(timestamp if timestamp is not None else 0)
            if u: # 如果是字符串，说明是user_id，如果是int，说明是re_id
                if type(u) == str:  # 如果是字符串，说明是user_id
                    user_id = u
                else:  # 如果是int，说明是re_id
                    user_id = self.indexer_u_rev[u] 
            if u and user_feat: # 如果用户特征不为空，则将用户特征插入到左侧（type=2）
                if type(u) == str:
                    u = 0
                if user_feat:
                    user_feat = self._process_cold_start_feat(user_feat) 
                ext_user_sequence.insert(0, (u, user_feat, 2, action_type, timestamp))  

            if i and item_feat: # 如果物品特征不为空，则将物品特征插入到右侧（type=1）
                # 序列对于训练时没见过的item，不会直接赋0，而是保留creative_id，creative_id远大于训练时的itemnum
                if i > self.itemnum:
                    i = 0
                if item_feat:
                    item_feat = self._process_cold_start_feat(item_feat) # 思考
                ext_user_sequence.append((i, item_feat, 1, action_type, timestamp)) 
    
                # 🕐 在启用时间特征且未隔离时间差特征时，为item交互计算时间间隔
                if self.enable_time_features and not self.disable_time_diff_features:
                    current_timestamp = timestamp if (timestamp is not None and timestamp > 0) else None
                    if last_item_timestamp is None:
                        # 第一个item：无前序上下文，使用默认值表示"序列开始/未知"
                        time_intervals.append((self.time_config['time_gap_buckets'], -1.0))  # (bucket_id, continuous_gap)
                    else:
                        if current_timestamp is not None and last_item_timestamp is not None:
                            # 计算与上一个item的时间间隔
                            time_delta = current_timestamp - last_item_timestamp
                            bucket_id = self._compute_time_interval_bucket(time_delta)
                            continuous_gap = self._compute_continuous_time_gap(time_delta)
                            time_intervals.append((bucket_id, continuous_gap))
                        else:
                            # 时间戳缺失，记作未知桶
                            time_intervals.append((self.time_config['time_gap_buckets'], -1.0))
                    # 更新最后一个item的有效时间戳（仅当本次有效时更新）
                    last_item_timestamp = current_timestamp if current_timestamp is not None else last_item_timestamp

        seq = np.zeros([self.maxlen + 1], dtype=np.int32)
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        seq_action_type = np.zeros([self.maxlen + 1], dtype=np.int32)  # 🎯 新增：历史token的动作类型
        seq_feat = np.empty([self.maxlen + 1], dtype=object)
        
        # 时间戳数组（用于时间感知注意力）
        seq_timestamps = np.zeros([self.maxlen + 1], dtype=np.int64)  # Unix时间戳
        
        # 收集所有需要的item_id，避免N+1查询，仅限于非预计算语义ID
        mm_emb_cache = {}
        if self.mm_emb_dict:
            all_needed_mm_items = set()
            for record_tuple in ext_user_sequence:
                i, feat, type_, action_type, timestamp = record_tuple  # 🔧 更新解构以匹配新结构
                if i != 0 and type_ == 1:  # 🔧 同步：只收集item类型的ID
                    all_needed_mm_items.add(i)
            # 第二阶段：批量预加载多模态特征到内存缓存
            mm_emb_cache = self._batch_load_multimodal_features(all_needed_mm_items)
        
        # 第三阶段：填充序列
        idx = self.maxlen
        time_interval_idx = len(time_intervals) - 1 if time_intervals else -1  
        for record_tuple in reversed(ext_user_sequence):
            i, feat, type_, action_type, timestamp = record_tuple  
            # 🎯 修正：当type_=2时，i才是user_id，而不是uid
            current_user_id = i if type_ == 2 else None  # 只有用户token才需要用户特征
            feat = self.fill_missing_feat_cached(feat, i, mm_emb_cache, include_time_features=True, user_id=current_user_id) # 返回的是一个完整的特征字典，先用默认值填充；key为feat_id，value为具体的值
            
            if self.enable_time_features and timestamp is not None:
                feat = dict(feat) if isinstance(feat, dict) else {}
                if (type_ == 1 and not self.disable_time_diff_features and 
                    time_interval_idx >= 0 and time_interval_idx < len(time_intervals)):
                    bucket_id, continuous_gap = time_intervals[time_interval_idx]
                    feat['time_gap'] = bucket_id
                    if self.time_config.get('enable_continuous_time_gap', True):
                        feat['time_gap_continuous'] = continuous_gap
                    time_interval_idx -= 1
                elif type_ == 1 and self.disable_time_diff_features:
                    if time_interval_idx >= 0:
                        time_interval_idx -= 1
                
                # 2. 添加扩展的绝对时间特征（所有token类型都需要，不受隔离开关影响）
                (hour_sin, hour_cos, weekday_sin, weekday_cos,
                 month_sin, month_cos, season_sin, season_cos,
                 day_of_year_sin, day_of_year_cos, is_weekend,
                 day_of_month_sin, day_of_month_cos) = self._compute_absolute_time_features(timestamp)
                
                # 分配到特征字典
                feat['hour_sin'] = hour_sin
                feat['hour_cos'] = hour_cos
                feat['weekday_sin'] = weekday_sin
                feat['weekday_cos'] = weekday_cos
                feat['month_sin'] = month_sin
                feat['month_cos'] = month_cos
                feat['season_sin'] = season_sin
                feat['season_cos'] = season_cos
                feat['day_of_year_sin'] = day_of_year_sin
                feat['day_of_year_cos'] = day_of_year_cos
                feat['is_weekend'] = is_weekend
                feat['day_of_month_sin'] = day_of_month_sin
                feat['day_of_month_cos'] = day_of_month_cos
            
            # 填充时间戳数组（用于时间感知注意力）
            if timestamp is not None and timestamp > 0:
                seq_timestamps[idx] = timestamp
            
            # 更新序列数据
            seq[idx] = i
            token_type[idx] = type_
            # 🎯 修复：action_type做+1映射避免与padding_idx=0冲突（与训练集保持一致）
            # 原始：0=曝光，1=点击；映射后：1=曝光，2=点击，0=PAD
            if action_type is not None:
                seq_action_type[idx] = action_type + 1  # 曝光0→1，点击1→2
            else:
                # 缺失action_type视为曝光（按用户需求）
                seq_action_type[idx] = 1  # 默认为曝光
            seq_feat[idx] = feat
            
            idx -= 1
            if idx == -1:
                break

        seq_feat = np.where(seq_feat == None, self.feature_default_value, seq_feat)

        # 转换为list便于处理，列表每一个元素是一个特征字典
        seq_feat_list = [seq_feat[i] for i in range(len(seq_feat))]
        
        # 处理特征张量化（与训练集MyDataset完全统一）
        feature_tensors = {}
        
        # 统一处理：与训练集保持完全一致的特征类型处理；思考：没有多模态特征？如果在RQVAE实时加载或者不用RQVAE
        all_feat_types = [
            ('item_sparse', self.feature_types.get('item_sparse', [])),
            ('item_array', self.feature_types.get('item_array', [])),
            ('item_continual', self.feature_types.get('item_continual', [])),
            ('user_sparse', self.feature_types.get('user_sparse', [])),
            ('user_array', self.feature_types.get('user_array', [])),
            ('user_continual', self.feature_types.get('user_continual', [])),
            ('seq_time_sparse', self.feature_types.get('seq_time_sparse', [])),
            ('seq_time_continual', self.feature_types.get('seq_time_continual', [])),  
            ('item_semantic_array', self.feature_types.get('item_semantic_array', [])),
        ]
                
        for feat_type, feat_ids in all_feat_types:
            for feat_id in feat_ids:
                # 针对单个batch，单个特征的numpy，数组是[maxlen,max_array_len]，其他是[maxlen]
                feature_tensors[f'seq_{feat_id}'] = self._feat2numpy_internal(seq_feat_list, feat_id)
        
        if not (self.enable_rqvae and self.use_precomputed_semantic_ids):
            # 端到端RQ-VAE模式或传统模式：处理原始多模态特征
            for feat_id in self.feature_types.get('item_emb', []):
                feature_tensors[f'seq_{feat_id}'] = self._process_multimodal_numpy(seq_feat_list, feat_id)

        # 添加时间戳
        feature_tensors['seq_timestamp'] = seq_timestamps
        feature_tensors['seq_action_type'] = seq_action_type

        return {
            'seq': seq,
            'token_type': token_type,
            'feature_tensors': feature_tensors,
            'user_id': user_id
        }

    def __len__(self):
        """用户数量（避免重复磁盘读取）"""
        return len(self.seq_offsets)

    @staticmethod
    def collate_fn(batch):
        """
        ⚡ 多进程优化版本：测试数据集的batch组装
        Args:
            batch: 多个__getitem__返回的数据字典

        Returns:
            测试batch的tensor字典
        """
        # 收集基础张量
        seq_list = [item['seq'] for item in batch]
        token_type_list = [item['token_type'] for item in batch]
        user_id_list = [item['user_id'] for item in batch]
        
        # 转换为batch tensor
        batch_data = {
            'seq': torch.from_numpy(np.array(seq_list)),
            'token_type': torch.from_numpy(np.array(token_type_list)),
            'user_id': user_id_list  # 保持为list，因为是字符串
        }
        
        # 收集特征张量，动态补齐array特征
        feature_tensors = {}
        if batch:  # 确保batch不为空
            # 获取所有特征键
            feature_keys = set()
            for item in batch:
                feature_keys.update(item['feature_tensors'].keys())
            
            # 为每个特征键构建batch tensor
            for feat_key in feature_keys:
                feat_batch = []
                for item in batch:
                    if feat_key in item['feature_tensors']:
                        feat_batch.append(item['feature_tensors'][feat_key])
                    else:
                        # 如果某个样本缺失该特征，用零填充
                        sample_feat = item['feature_tensors'][list(item['feature_tensors'].keys())[0]]
                        feat_batch.append(np.zeros_like(sample_feat))
                
                if feat_batch:
                    # 检查是否为array特征（2维且可能有不同长度）
                    first_tensor = feat_batch[0]
                    if first_tensor.ndim == 2 and first_tensor.shape[0] > 1:  # 可能是array特征
                        # 找到最大array长度
                        max_array_len = max(tensor.shape[-1] for tensor in feat_batch)
                        # 动态补齐
                        padded_batch = []
                        for tensor in feat_batch:
                            if tensor.shape[-1] < max_array_len:
                                pad_width = ((0, 0), (0, max_array_len - tensor.shape[-1]))
                                padded = np.pad(tensor, pad_width, mode='constant', constant_values=0)
                                padded_batch.append(padded)
                            else:
                                padded_batch.append(tensor)
                        feature_tensors[feat_key] = torch.from_numpy(np.stack(padded_batch))
                    else:
                        # 普通特征直接堆叠
                        feature_tensors[feat_key] = torch.from_numpy(np.array(feat_batch))
        
        batch_data['feature_tensors'] = feature_tensors
        return batch_data


def save_emb(emb, save_path):
    """
    将Embedding保存为二进制文件

    Args:
        emb: 要保存的Embedding，形状为 [num_points, num_dimensions]
        save_path: 保存路径
    """
    num_points = emb.shape[0]  # 数据点数量
    num_dimensions = emb.shape[1]  # 向量的维度
    print(f'saving {save_path}')
    with open(Path(save_path), 'wb') as f:
        f.write(struct.pack('II', num_points, num_dimensions))
        emb.tofile(f)


class OffsetMmEmbLoader:
    """
    🚀 完全照搬用户数据读取方式的多模态特征加载器
    核心思路：预计算文件偏移量，运行时直接seek读取
    """
    
    def __init__(self, mm_path, feat_ids, data_type="train"):
        self.mm_path = Path(mm_path)
        self.data_type = data_type
        self.offsets = {}  # {feat_id: {creative_id: (file_path, offset)}}
        
        for feat_id in feat_ids:
            if feat_id == '81':
                # 81特征使用原始pkl
                pkl_path = self.mm_path / f'emb_{feat_id}_32.pkl'
                if pkl_path.exists():
                    with open(pkl_path, 'rb') as f:
                        self.offsets[feat_id] = pickle.load(f)
                    print(f"✅ 加载81特征: {len(self.offsets[feat_id])} items")
            else:
                # 82等特征使用偏移量文件 - 🚀 从多个位置搜索文件，优先使用data_type特定文件
                offset_file_found = False
                search_paths = []
                
                # 搜索路径：cache_dir > ckpt_dir > 原始路径 > 当前目录
                cache_dir = os.environ.get('USER_CACHE_PATH')
                ckpt_dir = os.environ.get('TRAIN_CKPT_PATH')
                
                # 优先搜索包含data_type的新格式文件
                if cache_dir:
                    search_paths.append(Path(cache_dir) / f'mm_emb_{feat_id}_{self.data_type}_offsets.pkl')
                    search_paths.append(Path(cache_dir) / f'mm_emb_{feat_id}_offsets.pkl')  # 兼容旧格式
                if ckpt_dir:
                    search_paths.append(Path(ckpt_dir, f"global_step_final") / f'mm_emb_{feat_id}_{self.data_type}_offsets.pkl')
                    search_paths.append(Path(ckpt_dir, f"global_step_final") / f'mm_emb_{feat_id}_offsets.pkl')  # 兼容旧格式
                search_paths.append(self.mm_path / f'mm_emb_{feat_id}_{self.data_type}_offsets.pkl')  # 原始路径新格式
                search_paths.append(self.mm_path / f'mm_emb_{feat_id}_offsets.pkl')  # 原始路径旧格式
                search_paths.append(Path('.') / f'mm_emb_{feat_id}_{self.data_type}_offsets.pkl')  # 当前目录新格式
                search_paths.append(Path('.') / f'mm_emb_{feat_id}_offsets.pkl')  # 当前目录旧格式
                
                for offset_file in search_paths:
                    if offset_file.exists():
                        with open(offset_file, 'rb') as f:
                            self.offsets[feat_id] = pickle.load(f)
                        print(f"✅ 加载{feat_id}偏移量索引: {len(self.offsets[feat_id])} items (from {offset_file})")
                        offset_file_found = True
                        break
                
                if not offset_file_found:
                    print(f"❌ 未找到{feat_id}偏移量文件，搜索路径:")
                    for path in search_paths:
                        print(f"   - {path}")
                    print(f"   请先运行: python build_mm_emb_offsets.py")
    
    def _load_mm_emb(self, feat_id, creative_id):
        """完全照搬 _load_user_data 的逻辑"""
        if feat_id not in self.offsets or creative_id not in self.offsets[feat_id]:
            return None
        
        if feat_id == '81':
            # 81是字典形式
            return self.offsets[feat_id].get(creative_id)
        else:
            # 82等是文件偏移形式
            file_path, offset = self.offsets[feat_id][creative_id]
            try:
                with open(file_path, 'rb') as f:
                    f.seek(offset)
                    line = f.readline()
                    data = json.loads(line.decode('utf-8').strip())
                    emb = data['emb']
                    if isinstance(emb, list):
                        emb = np.array(emb, dtype=np.float32)
                    return emb
            except Exception as e:
                return None
    
    def get(self, feat_id, creative_id):
        """获取多模态向量"""
        return self._load_mm_emb(feat_id, creative_id)

def load_mm_emb(mm_path, feat_ids, data_type="train"):
    """
    🚀 使用偏移量方式加载多模态特征 - 完全照搬用户数据读取模式
    
    Args:
        mm_path: 多模态数据路径
        feat_ids: 特征ID列表
        data_type: 数据类型 ("train" 或 "eval")，用于选择对应的offset文件
    """
    return OffsetMmEmbLoader(mm_path, feat_ids, data_type)