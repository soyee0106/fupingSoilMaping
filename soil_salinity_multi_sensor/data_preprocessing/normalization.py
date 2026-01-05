"""
数据归一化模块

提供数据标准化和归一化功能，支持保存和加载参数，
确保训练和预测数据的一致性处理。
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Optional, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StandardScaler:
    """
    标准化类，使用Z-score标准化：x' = (x - mean) / std
    """
    
    def __init__(self):
        """初始化标准化器。"""
        self.mean_ = None
        self.std_ = None
        self.fitted = False
    
    def fit(self, X: np.ndarray):
        """
        拟合标准化器，计算均值和标准差。
        
        参数:
            X: 输入数据，形状为 (n_samples, n_features)
        """
        self.mean_ = np.mean(X, axis=0, keepdims=True)
        self.std_ = np.std(X, axis=0, keepdims=True)
        # 避免除零
        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)
        self.fitted = True
        logger.info(f"StandardScaler fitted: mean shape={self.mean_.shape}, std shape={self.std_.shape}")
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        对数据进行标准化变换。
        
        参数:
            X: 输入数据
        
        返回:
            标准化后的数据
        """
        if not self.fitted:
            raise ValueError("Scaler has not been fitted yet. Call fit() first.")
        
        return (X - self.mean_) / self.std_
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        拟合并变换数据。
        
        参数:
            X: 输入数据
        
        返回:
            标准化后的数据
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        逆变换，将标准化后的数据还原。
        
        参数:
            X: 标准化后的数据
        
        返回:
            原始尺度的数据
        """
        if not self.fitted:
            raise ValueError("Scaler has not been fitted yet. Call fit() first.")
        
        return X * self.std_ + self.mean_
    
    def save(self, filepath: Union[str, Path]):
        """
        保存标准化器参数。
        
        参数:
            filepath: 保存路径
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'mean_': self.mean_,
                'std_': self.std_,
                'fitted': self.fitted
            }, f)
        
        logger.info(f"StandardScaler saved to {filepath}")
    
    def load(self, filepath: Union[str, Path]):
        """
        加载标准化器参数。
        
        参数:
            filepath: 加载路径
        """
        filepath = Path(filepath)
        
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
        
        self.mean_ = params['mean_']
        self.std_ = params['std_']
        self.fitted = params['fitted']
        
        logger.info(f"StandardScaler loaded from {filepath}")


class MinMaxScaler:
    """
    最小-最大归一化类：x' = (x - min) / (max - min)
    将数据缩放到[0, 1]区间。
    """
    
    def __init__(self, feature_range: tuple = (0, 1)):
        """
        初始化归一化器。
        
        参数:
            feature_range: 目标范围，默认为(0, 1)
        """
        self.feature_range = feature_range
        self.min_ = None
        self.max_ = None
        self.scale_ = None
        self.fitted = False
    
    def fit(self, X: np.ndarray):
        """
        拟合归一化器，计算最小值和最大值。
        
        参数:
            X: 输入数据，形状为 (n_samples, n_features)
        """
        self.min_ = np.min(X, axis=0, keepdims=True)
        self.max_ = np.max(X, axis=0, keepdims=True)
        # 计算缩放因子
        data_range = self.max_ - self.min_
        # 避免除零
        data_range = np.where(data_range == 0, 1.0, data_range)
        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / data_range
        self.fitted = True
        logger.info(f"MinMaxScaler fitted: min shape={self.min_.shape}, max shape={self.max_.shape}")
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        对数据进行归一化变换。
        
        参数:
            X: 输入数据
        
        返回:
            归一化后的数据
        """
        if not self.fitted:
            raise ValueError("Scaler has not been fitted yet. Call fit() first.")
        
        X_scaled = (X - self.min_) * self.scale_ + self.feature_range[0]
        return X_scaled
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        拟合并变换数据。
        
        参数:
            X: 输入数据
        
        返回:
            归一化后的数据
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        逆变换，将归一化后的数据还原。
        
        参数:
            X: 归一化后的数据
        
        返回:
            原始尺度的数据
        """
        if not self.fitted:
            raise ValueError("Scaler has not been fitted yet. Call fit() first.")
        
        X_original = (X - self.feature_range[0]) / self.scale_ + self.min_
        return X_original
    
    def save(self, filepath: Union[str, Path]):
        """
        保存归一化器参数。
        
        参数:
            filepath: 保存路径
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'min_': self.min_,
                'max_': self.max_,
                'scale_': self.scale_,
                'feature_range': self.feature_range,
                'fitted': self.fitted
            }, f)
        
        logger.info(f"MinMaxScaler saved to {filepath}")
    
    def load(self, filepath: Union[str, Path]):
        """
        加载归一化器参数。
        
        参数:
            filepath: 加载路径
        """
        filepath = Path(filepath)
        
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
        
        self.min_ = params['min_']
        self.max_ = params['max_']
        self.scale_ = params['scale_']
        self.feature_range = params['feature_range']
        self.fitted = params['fitted']
        
        logger.info(f"MinMaxScaler loaded from {filepath}")

