"""
数据预处理工具函数

提供掩码向量解析等工具函数。
"""

import numpy as np
import pandas as pd
from typing import Optional


def parse_band_mask(mask_str: str) -> np.ndarray:
    """
    解析掩码字符串为numpy数组。
    
    参数:
        mask_str: 掩码字符串，格式如 "1,1,0,1"（逗号分隔）
    
    返回:
        掩码数组，形状为 (n_bands,)
    """
    if isinstance(mask_str, str):
        return np.array([float(x) for x in mask_str.split(',')])
    else:
        return np.array(mask_str)


def get_band_mask_from_df(df: pd.DataFrame, mask_column: str = 'band_mask') -> np.ndarray:
    """
    从DataFrame中提取掩码向量。
    
    参数:
        df: 包含掩码列的DataFrame
        mask_column: 掩码列名
    
    返回:
        掩码数组，形状为 (n_samples, n_bands)
    """
    if mask_column not in df.columns:
        # 如果没有掩码列，返回全1掩码（所有波段都有效）
        n_samples = len(df)
        return np.ones((n_samples, 4))
    
    masks = []
    for mask_str in df[mask_column]:
        mask = parse_band_mask(mask_str)
        masks.append(mask)
    
    return np.array(masks)

