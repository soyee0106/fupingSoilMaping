"""
训练模块

该模块包含所有模型训练相关的功能：
- 阶段一预训练：训练传感器偏差解码器
- 阶段二预训练：训练盐分反演网络
- 联合微调：端到端微调完整模型
- 基线模型训练：训练Model A和Model B
"""

from .pretrain_stage1 import train_stage1_decoder
from .pretrain_stage2 import train_stage2_inverter
from .joint_finetune import fine_tune_full_model
from .train_baselines import train_baseline_a, train_baseline_b

__all__ = [
    'train_stage1_decoder',
    'train_stage2_inverter',
    'fine_tune_full_model',
    'train_baseline_a',
    'train_baseline_b',
]

