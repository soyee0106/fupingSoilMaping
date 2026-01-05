"""
数据预处理模块

该模块包含数据预处理相关的所有功能：
- 波段匹配：定义无人机波段与卫星波段的对应关系
- 光谱指数计算：计算各种植被和土壤指数
- 数据配对：从配准后的影像中提取样点光谱值
- 数据归一化：标准化和归一化处理
"""

from .band_matching import create_band_mapping
from .spectral_indices import (
    calculate_ndvi,
    calculate_si,
    calculate_ndsi,
    calculate_evi,
    calculate_all_indices
)
from .data_pairing import DataPairer
from .normalization import StandardScaler, MinMaxScaler
from .utils import parse_band_mask, get_band_mask_from_df

__all__ = [
    'create_band_mapping',
    'calculate_ndvi',
    'calculate_si',
    'calculate_ndsi',
    'calculate_evi',
    'calculate_all_indices',
    'DataPairer',
    'StandardScaler',
    'MinMaxScaler',
    'parse_band_mask',
    'get_band_mask_from_df',
]

