#!/bin/bash

# show ${RUNTIME_SCRIPT_DIR}
echo ${RUNTIME_SCRIPT_DIR}
# enter train workspace
cd ${RUNTIME_SCRIPT_DIR}

# 设置单机分布式所需的环境变量，避免 Muon 初始化报错
export MASTER_ADDR=localhost
export MASTER_PORT=12355
pip install muon_optimizer-0.1.0-py3-none-any.whl

python -u main.py \
  --enable_rqvae \
  --use_precomputed_semantic_ids \
  --skip_rqvae_training \
  --rqvae_features 81 82 84 86\
  --rqvae_epochs 5 \
  --rqvae_batch_size 1024 \
  --rqvae_lr 0.002 \
  --mm_emb_id 81 82 84 \
  --hidden_units 512 \
  --MLP_dropout_rate 0.05 \
  --emb_dropout_rate 0.05 \
  --transformer_dropout 0.05 \
  --id_dropout_rate 0.02 \
  --exposure_weight_start 0.4 \
  --exposure_weight_end 0.01 \
  --exposure_decay_strategy cosine \
  --lr_schedule cosine \
  --min_lr_ratio 0.01 \
  --num_blocks 16 \
  --num_epochs 10 \
  --num_heads 8 \
  --batch_size 256 \
  --lr 0.002 \
  --warmup_steps 200 \
  --max_grad_norm 1.0 \
  --weight_decay 0.01 \
  --temperature 0.03 \
  --use_muon \
  --muon_lr 0.02 \
  --muon_aux_lr 0.004 \
  --attention_mode hstu \
  # --enable_relative_bias \
  --enable_time_features \
  --enable_time_bias \
  # --disable_time_diff_features \
  --enable_chunked_computation \
  --infonce_chunk_size 512 \
  # === Action-aware margin (启用以提升点击导向指标) ===
  --enable_action_margin \
  --action_margin_click 0.0 \
  --action_margin_exposure 0.0 \
  --norm_first \
  # --enable_popularity_sampling \
  # --enable_alias_sampling \
  # --enable_false_negative_filter \
  --enable_ctr_feature \
  --enable_mixed_precision \
  # --full_train \
  --enable_rope \
  --rope_theta 10000.0 \
  --rope_max_seq_len 128 \
  --item_cand_head linear \
  #--state_dict_path global_step27125.valid_loss=2.9757 \
  #--enable_inbatch_negatives 