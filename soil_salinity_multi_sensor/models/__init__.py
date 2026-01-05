"""
模型定义模块

该模块包含所有深度学习模型的定义：
- Stage1: 差异解耦网络（传感器偏差解码器）
- Stage2: 盐分反演网络
- FullModelC: 完整的两阶段模型
- BaselineModelA: 基线模型A（卫星直接到盐分）
- BaselineModelB: 基线模型B（卫星到无人机波段再到盐分）
"""

from .stage1_decoder import SensorBiasDecoder
from .stage2_inverter import SalinityInverter
from .full_model import FullModelC
from .baseline_models import BaselineModelA, BaselineModelB

__all__ = [
    'SensorBiasDecoder',
    'SalinityInverter',
    'FullModelC',
    'BaselineModelA',
    'BaselineModelB',
]

