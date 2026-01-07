"""
推理模块

提供模型推理功能，用于生成预测结果。
"""

from .stage1_inference import run_stage1_inference, predict_raster

__all__ = ['run_stage1_inference', 'predict_raster']

