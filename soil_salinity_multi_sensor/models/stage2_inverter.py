"""
阶段二：盐分反演网络

该网络从模拟的无人机波段反射率和光谱指数中预测土壤盐分值。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SalinityInverter(nn.Module):
    """
    盐分反演网络（阶段二网络）
    
    功能：从无人机波段反射率和光谱指数中预测土壤盐分值。
    
    输入：
        - 无人机4个波段反射率（G, R, REG, NIR）
        - 由这些波段计算的光谱指数（28个指数：S1, S1REG, NDSI, NDSIREG, SI1, SI1REG等）
    
    输出：
        - 土壤盐分值（标量）
    """
    
    def __init__(
        self,
        n_uav_bands: int = 4,
        n_spectral_indices: int = 26,
        hidden_dims: list = [64, 32, 16],
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True
    ):
        """
        初始化盐分反演网络。
        
        参数:
            n_uav_bands: 无人机波段数量（默认4个：G, R, REG, NIR）
            n_spectral_indices: 光谱指数数量（默认26个）
            hidden_dims: 隐藏层维度列表，如[64, 32, 16]
            dropout_rate: Dropout比率
            use_batch_norm: 是否使用批归一化
        """
        super(SalinityInverter, self).__init__()
        
        self.n_uav_bands = n_uav_bands
        self.n_spectral_indices = n_spectral_indices
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # 输入特征维度 = 无人机波段 + 光谱指数
        input_dim = n_uav_bands + n_spectral_indices
        
        # 构建全连接层
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # 输出层：1个标量（盐分值）
        # 回归任务，不使用激活函数
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重。"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        uav_bands: torch.Tensor,
        spectral_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播。
        
        参数:
            uav_bands: 无人机波段反射率，形状为 (batch_size, n_uav_bands)
            spectral_indices: 光谱指数，形状为 (batch_size, n_spectral_indices)
        
        返回:
            预测的盐分值，形状为 (batch_size, 1)
        """
        # 拼接输入特征
        x = torch.cat([uav_bands, spectral_indices], dim=1)
        
        # 通过网络
        salinity = self.network(x)
        
        return salinity
    
    def get_feature_dim(self) -> int:
        """返回输入特征的总维度。"""
        return self.n_uav_bands + self.n_spectral_indices

