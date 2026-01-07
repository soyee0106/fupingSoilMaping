"""
标准化工具函数

用于在模型推理时加载和使用标准化器。
"""

from pathlib import Path
from typing import Dict, Optional
import numpy as np
import pandas as pd
import rasterio
from .normalization import StandardScaler
import logging

logger = logging.getLogger(__name__)


def load_scalers(scalers_dir: Path) -> Dict[str, StandardScaler]:
    """
    加载所有标准化器。
    
    参数:
        scalers_dir: 标准化器参数目录
    
    返回:
        字典，包含 's2', 'l8', 'uav' 三个标准化器
    """
    scalers_dir = Path(scalers_dir)
    
    scalers = {}
    
    # 加载S2标准化器
    s2_path = scalers_dir / 'scaler_s2.pkl'
    if s2_path.exists():
        scaler_s2 = StandardScaler()
        scaler_s2.load(s2_path)
        scalers['s2'] = scaler_s2
        logger.info(f"Loaded S2 scaler from {s2_path}")
    else:
        logger.warning(f"S2 scaler not found at {s2_path}")
    
    # 加载L8标准化器
    l8_path = scalers_dir / 'scaler_l8.pkl'
    if l8_path.exists():
        scaler_l8 = StandardScaler()
        scaler_l8.load(l8_path)
        scalers['l8'] = scaler_l8
        logger.info(f"Loaded L8 scaler from {l8_path}")
    else:
        logger.warning(f"L8 scaler not found at {l8_path}")
    
    # 加载UAV标准化器
    uav_path = scalers_dir / 'scaler_uav.pkl'
    if uav_path.exists():
        scaler_uav = StandardScaler()
        scaler_uav.load(uav_path)
        scalers['uav'] = scaler_uav
        logger.info(f"Loaded UAV scaler from {uav_path}")
    else:
        logger.warning(f"UAV scaler not found at {uav_path}")
    
    return scalers


def normalize_satellite_data(
    data: np.ndarray,
    sensor_id: int,
    scalers: Dict[str, StandardScaler],
    sensor_labels: Dict[str, int]
) -> np.ndarray:
    """
    标准化卫星数据。
    
    参数:
        data: 卫星数据，形状为 (n_samples, n_bands) 或 (height, width, n_bands)
        sensor_id: 传感器ID（0=S2, 1=L8）
        scalers: 标准化器字典
        sensor_labels: 传感器标签字典
    
    返回:
        标准化后的数据
    """
    # 确定使用哪个标准化器
    if sensor_id == sensor_labels.get('S2', 0):
        scaler = scalers.get('s2')
        sensor_name = 'S2'
    elif sensor_id == sensor_labels.get('L8', 1):
        scaler = scalers.get('l8')
        sensor_name = 'L8'
    else:
        raise ValueError(f"Unknown sensor_id: {sensor_id}")
    
    if scaler is None:
        raise ValueError(f"Scaler for {sensor_name} not found!")
    
    # 处理不同形状的输入
    original_shape = data.shape
    if len(original_shape) == 3:
        # 影像格式 (height, width, bands) -> (height*width, bands)
        data_2d = data.reshape(-1, original_shape[2])
    elif len(original_shape) == 2:
        # 已经是2D格式 (n_samples, n_bands)
        data_2d = data
    else:
        raise ValueError(f"Unsupported data shape: {original_shape}")
    
    # 标准化
    data_normalized = scaler.transform(data_2d)
    
    # 恢复原始形状
    if len(original_shape) == 3:
        data_normalized = data_normalized.reshape(original_shape)
    
    return data_normalized


def normalize_uav_data(
    data: np.ndarray,
    scalers: Dict[str, StandardScaler]
) -> np.ndarray:
    """
    标准化UAV数据。
    
    参数:
        data: UAV数据，形状为 (n_samples, n_bands) 或 (height, width, n_bands)
        scalers: 标准化器字典
    
    返回:
        标准化后的数据
    """
    scaler = scalers.get('uav')
    if scaler is None:
        raise ValueError("UAV scaler not found!")
    
    # 处理不同形状的输入
    original_shape = data.shape
    if len(original_shape) == 3:
        # 影像格式 (height, width, bands) -> (height*width, bands)
        data_2d = data.reshape(-1, original_shape[2])
    elif len(original_shape) == 2:
        # 已经是2D格式 (n_samples, n_bands)
        data_2d = data
    else:
        raise ValueError(f"Unsupported data shape: {original_shape}")
    
    # 标准化
    data_normalized = scaler.transform(data_2d)
    
    # 恢复原始形状
    if len(original_shape) == 3:
        data_normalized = data_normalized.reshape(original_shape)
    
    return data_normalized


def normalize_raster(
    raster_path: Path,
    output_path: Path,
    scaler: StandardScaler,
    band_indices: Optional[list] = None
) -> None:
    """
    标准化栅格影像。
    
    参数:
        raster_path: 输入栅格路径
        output_path: 输出栅格路径
        scaler: 标准化器
        band_indices: 要标准化的波段索引（从1开始），如果为None则处理所有波段
    """
    with rasterio.open(raster_path) as src:
        if band_indices is None:
            band_indices = list(range(1, src.count + 1))
        
        # 读取所有波段
        bands = []
        for idx in band_indices:
            band = src.read(idx)
            bands.append(band)
        
        # 标准化每个波段
        bands_normalized = []
        for i, band in enumerate(bands):
            band_flat = band.flatten()
            valid_mask = ~np.isnan(band_flat)
            
            if valid_mask.any():
                # 使用对应波段的标准化参数
                if i < scaler.mean_.shape[1]:
                    band_mean = scaler.mean_[0, i]
                    band_std = scaler.std_[0, i]
                else:
                    band_mean = scaler.mean_[0, -1]
                    band_std = scaler.std_[0, -1]
                
                band_normalized = band_flat.copy()
                band_normalized[valid_mask] = (band_flat[valid_mask] - band_mean) / band_std
            else:
                band_normalized = band_flat
            
            bands_normalized.append(band_normalized.reshape(band.shape))
        
        # 保存标准化后的影像
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=src.height,
            width=src.width,
            count=len(bands_normalized),
            dtype='float32',
            crs=src.crs,
            transform=src.transform,
            nodata=None,
            compress='lzw'
        ) as dst:
            for i, band in enumerate(bands_normalized, 1):
                dst.write(band, i)
        
        logger.info(f"Normalized raster saved to {output_path}")

