"""
PyTorch数据集类

用于加载预处理后的数据，支持不同的模型训练需求。
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DenseMappingDataset(Dataset):
    """
    密集映射数据集（用于阶段一训练：学习从卫星光谱到无人机光谱的映射）
    
    输入：卫星波段 + 光谱指数 + 传感器ID + 掩码向量
    输出：无人机波段
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        satellite_band_cols: List[str],
        uav_band_cols: List[str],
        sensor_id_col: str = 'sensor_id',
        band_mask: Optional[np.ndarray] = None,
        normalize: bool = True,
        scaler: Optional[object] = None,
        calculate_indices: bool = True,
        data_already_normalized: bool = False
    ):
        """
        初始化数据集。
        
        参数:
            df: 包含卫星和无人机波段数据的DataFrame
            satellite_band_cols: 卫星波段列名列表（如['SAT_band_1', 'SAT_band_2', ...]）
            uav_band_cols: 无人机波段列名列表（如['UAV_band_1', 'UAV_band_2', ...]）
            sensor_id_col: 传感器ID列名（默认'sensor_id'）
            band_mask: 波段掩码向量（如[1, 1, 0, 1]表示REG波段缺失）
            normalize: 是否归一化（如果data_already_normalized=True，此参数无效）
            scaler: 归一化器（如果提供，使用它；否则创建新的）
            calculate_indices: 是否计算光谱指数
            data_already_normalized: 数据是否已经标准化（如果True，跳过标准化步骤）
        """
        self.df = df.copy()
        self.satellite_band_cols = satellite_band_cols
        self.uav_band_cols = uav_band_cols
        self.sensor_id_col = sensor_id_col
        self.data_already_normalized = data_already_normalized
        # 如果数据已经标准化，就不再进行标准化
        self.normalize = normalize and not data_already_normalized
        self.calculate_indices = calculate_indices
        
        # 提取卫星波段数据（顺序：G, R, REG, NIR）
        self.satellite_bands = self.df[satellite_band_cols].values.astype(np.float32)
        
        # 提取传感器ID
        if sensor_id_col in self.df.columns:
            self.sensor_ids = self.df[sensor_id_col].values.astype(np.int64)
        else:
            logger.warning(f"Column '{sensor_id_col}' not found, using default sensor_id=0")
            self.sensor_ids = np.zeros(len(self.df), dtype=np.int64)
        
        # 提取无人机波段数据
        self.uav_bands = self.df[uav_band_cols].values.astype(np.float32)
        
        # 处理缺失值（NaN或空值）
        valid_mask = ~(np.isnan(self.satellite_bands).any(axis=1) | 
                      np.isnan(self.uav_bands).any(axis=1))
        self.satellite_bands = self.satellite_bands[valid_mask]
        self.uav_bands = self.uav_bands[valid_mask]
        self.sensor_ids = self.sensor_ids[valid_mask]
        
        logger.info(f"Valid samples: {len(self.satellite_bands)} / {len(df)}")
        
        # 计算光谱指数（从卫星波段）
        if calculate_indices:
            from data_preprocessing.spectral_indices import calculate_all_indices
            
            # 将波段数据转换为字典格式（G, R, REG, NIR）
            # 假设顺序是：G, R, REG, NIR
            band_dict = {
                'G': self.satellite_bands[:, 0],
                'R': self.satellite_bands[:, 1],
                'REG': self.satellite_bands[:, 2],
                'NIR': self.satellite_bands[:, 3]
            }
            
            # 计算所有26个指数
            indices_dict = calculate_all_indices(band_dict)
            
            # 将指数转换为数组（按固定顺序）
            index_order = [
                'S1', 'S1REG', 'NDSI', 'NDSIREG', 'SI1', 'SI1REG', 'SI2', 'SI2REG',
                'SI3', 'SI3REG', 'SIT', 'SITREG', 'Int1', 'Int1REG', 'Int2', 'Int2REG',
                'NDVI', 'NDVIREG', 'SAVI', 'SAVIREG', 'BI', 'BIREG', 'IFe2O3', 'IFe2O3REG',
                'DVI', 'DVIREG'
            ]
            
            self.spectral_indices = np.column_stack([
                indices_dict[idx] for idx in index_order
            ]).astype(np.float32)
            
            # 处理指数中的NaN值（用0填充）
            self.spectral_indices = np.nan_to_num(self.spectral_indices, nan=0.0, posinf=0.0, neginf=0.0)
            
            logger.info(f"Calculated {len(index_order)} spectral indices")
        else:
            self.spectral_indices = None
        
        # 设置波段掩码（根据传感器ID动态生成）
        # S2: [1, 1, 1, 1], L8: [1, 1, 0, 1]
        self.band_masks = np.zeros((len(self.satellite_bands), len(satellite_band_cols)), dtype=np.float32)
        for i, sensor_id in enumerate(self.sensor_ids):
            if sensor_id == 0:  # S2
                self.band_masks[i] = [1.0, 1.0, 1.0, 1.0]
            elif sensor_id == 1:  # L8
                self.band_masks[i] = [1.0, 1.0, 0.0, 1.0]
            else:
                # 默认使用提供的band_mask或全1
                if band_mask is not None:
                    self.band_masks[i] = band_mask
                else:
                    self.band_masks[i] = [1.0, 1.0, 1.0, 1.0]
        
        # 归一化卫星波段（但不归一化指数）
        # 如果数据已经标准化，跳过此步骤
        if self.normalize:
            if scaler is None:
                from sklearn.preprocessing import StandardScaler
                self.scaler = StandardScaler()
                self.satellite_bands = self.scaler.fit_transform(self.satellite_bands)
            else:
                self.scaler = scaler
                self.satellite_bands = self.scaler.transform(self.satellite_bands)
        else:
            self.scaler = None
            if data_already_normalized:
                logger.info("数据已经标准化，跳过标准化步骤")
    
    def __len__(self):
        return len(self.satellite_bands)
    
    def __getitem__(self, idx):
        """
        返回一个样本。
        
        返回:
            satellite_bands: 卫星波段值（已归一化）
            spectral_indices: 光谱指数
            sensor_onehot: 传感器独热编码
            band_mask: 波段掩码向量
            uav_bands: 无人机波段值（目标）
        """
        satellite = torch.FloatTensor(self.satellite_bands[idx])
        uav = torch.FloatTensor(self.uav_bands[idx])
        sensor_id = self.sensor_ids[idx]
        
        # 传感器独热编码（2个传感器：S2=0, L8=1）
        sensor_onehot = torch.zeros(2, dtype=torch.float32)
        sensor_onehot[sensor_id] = 1.0
        
        mask = torch.FloatTensor(self.band_masks[idx])
        
        if self.spectral_indices is not None:
            indices = torch.FloatTensor(self.spectral_indices[idx])
        else:
            indices = torch.zeros(26, dtype=torch.float32)  # 默认26个指数
        
        return {
            'satellite_bands': satellite,
            'spectral_indices': indices,
            'sensor_onehot': sensor_onehot,
            'band_mask': mask,
            'uav_bands': uav
        }


class SalinityDataset(Dataset):
    """
    盐分反演数据集（用于阶段二训练：从无人机光谱预测盐分值）
    
    输入：无人机波段 + 光谱指数
    输出：盐分值
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        uav_band_cols: List[str],
        spectral_index_cols: List[str],
        salinity_col: str,
        normalize: bool = True,
        scaler: Optional[object] = None
    ):
        """
        初始化数据集。
        
        参数:
            df: 包含无人机波段、光谱指数和盐分值的DataFrame
            uav_band_cols: 无人机波段列名列表
            spectral_index_cols: 光谱指数列名列表
            salinity_col: 盐分值列名
            normalize: 是否归一化
            scaler: 归一化器
        """
        self.df = df.copy()
        self.uav_band_cols = uav_band_cols
        self.spectral_index_cols = spectral_index_cols
        self.salinity_col = salinity_col
        self.normalize = normalize
        
        # 提取特征（波段 + 指数）
        feature_cols = uav_band_cols + spectral_index_cols
        self.features = self.df[feature_cols].values.astype(np.float32)
        
        # 提取盐分值
        self.salinity = self.df[salinity_col].values.astype(np.float32)
        
        # 处理缺失值
        valid_mask = ~(np.isnan(self.features).any(axis=1) | np.isnan(self.salinity))
        self.features = self.features[valid_mask]
        self.salinity = self.salinity[valid_mask]
        
        logger.info(f"Valid samples: {len(self.features)} / {len(df)}")
        
        # 归一化
        if normalize:
            if scaler is None:
                from sklearn.preprocessing import StandardScaler
                self.scaler = StandardScaler()
                self.features = self.scaler.fit_transform(self.features)
            else:
                self.scaler = scaler
                self.features = self.scaler.transform(self.features)
        else:
            self.scaler = None
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        """
        返回一个样本。
        
        返回:
            features: 特征向量（波段 + 指数）
            salinity: 盐分值
        """
        features = torch.FloatTensor(self.features[idx])
        salinity = torch.FloatTensor([self.salinity[idx]])
        
        return {
            'features': features,
            'salinity': salinity
        }


class FullModelDataset(Dataset):
    """
    完整模型数据集（用于联合微调或基线模型）
    
    输入：卫星波段 + 传感器ID + 掩码向量
    输出：盐分值
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        satellite_band_cols: List[str],
        sensor_id: int = 0,
        band_mask: Optional[np.ndarray] = None,
        salinity_col: Optional[str] = None,
        normalize: bool = True,
        scaler: Optional[object] = None
    ):
        """
        初始化数据集。
        
        参数:
            df: 包含卫星波段和盐分值的DataFrame
            satellite_band_cols: 卫星波段列名列表
            sensor_id: 传感器ID
            band_mask: 波段掩码向量
            salinity_col: 盐分值列名（如果为None，则不包含盐分值，用于预测）
            normalize: 是否归一化
            scaler: 归一化器
        """
        self.df = df.copy()
        self.satellite_band_cols = satellite_band_cols
        self.sensor_id = sensor_id
        self.normalize = normalize
        self.has_salinity = salinity_col is not None
        
        # 提取卫星波段
        self.satellite_bands = self.df[satellite_band_cols].values.astype(np.float32)
        
        # 提取盐分值（如果有）
        if self.has_salinity:
            self.salinity = self.df[salinity_col].values.astype(np.float32)
            # 处理缺失值
            valid_mask = ~(np.isnan(self.satellite_bands).any(axis=1) | np.isnan(self.salinity))
            self.satellite_bands = self.satellite_bands[valid_mask]
            self.salinity = self.salinity[valid_mask]
        else:
            # 只处理卫星波段的缺失值
            valid_mask = ~np.isnan(self.satellite_bands).any(axis=1)
            self.satellite_bands = self.satellite_bands[valid_mask]
            self.salinity = None
        
        logger.info(f"Valid samples: {len(self.satellite_bands)} / {len(df)}")
        
        # 设置波段掩码
        if band_mask is None:
            self.band_mask = np.ones(len(satellite_band_cols), dtype=np.float32)
        else:
            self.band_mask = np.array(band_mask, dtype=np.float32)
        
        # 归一化
        if normalize:
            if scaler is None:
                from sklearn.preprocessing import StandardScaler
                self.scaler = StandardScaler()
                self.satellite_bands = self.scaler.fit_transform(self.satellite_bands)
            else:
                self.scaler = scaler
                self.satellite_bands = self.scaler.transform(self.satellite_bands)
        else:
            self.scaler = None
    
    def __len__(self):
        return len(self.satellite_bands)
    
    def __getitem__(self, idx):
        """
        返回一个样本。
        """
        satellite = torch.FloatTensor(self.satellite_bands[idx])
        sensor = torch.LongTensor([self.sensor_id])
        mask = torch.FloatTensor(self.band_mask)
        
        result = {
            'satellite_bands': satellite,
            'sensor_id': sensor,
            'band_mask': mask
        }
        
        if self.has_salinity:
            result['salinity'] = torch.FloatTensor([self.salinity[idx]])
        
        return result

