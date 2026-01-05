"""
阶段一：差异解耦网络（传感器偏差解码器）

该网络学习并校正不同卫星（Sentinel-2, Landsat-8）的光谱偏差，
将卫星光谱特征转换为模拟的无人机4个波段反射率。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SensorBiasDecoder(nn.Module):
    """
    传感器偏差解码器（阶段一网络）
    
    功能：将卫星光谱特征（波段+指数）和传感器标识转换为模拟的无人机波段反射率。
    支持掩码机制处理波段缺失（如L8没有REG波段）。
    
    输入：
        - 卫星波段特征（统一为G, R, REG, NIR格式，缺失位置填充0）
        - 光谱指数特征（26个指数）
        - 传感器独热编码标签（S2或L8）
        - 波段掩码向量（标记哪些波段有效：1=有效，0=缺失）
    
    输出：
        - 模拟的无人机4个波段反射率（G, R, REG, NIR）
    """
    
    def __init__(
        self,
        n_satellite_bands: int = 4,
        n_spectral_indices: int = 26,
        n_sensors: int = 2,
        n_mask_bits: int = 4,  # 掩码位数（对应4个波段：G, R, REG, NIR）
        hidden_dims: list = [128, 64, 32],
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True,
        use_mask: bool = True  # 是否使用掩码机制
    ):
        """
        初始化传感器偏差解码器。
        
        参数:
            n_satellite_bands: 卫星波段数量（默认4个：G, R, REG, NIR）
            n_spectral_indices: 光谱指数数量（默认26个）
            n_sensors: 传感器数量（默认2个：S2, L8）
            n_mask_bits: 掩码位数（默认4，对应4个波段）
            hidden_dims: 隐藏层维度列表，如[128, 64, 32]
            dropout_rate: Dropout比率
            use_batch_norm: 是否使用批归一化
            use_mask: 是否使用掩码机制
        """
        super(SensorBiasDecoder, self).__init__()
        
        self.n_satellite_bands = n_satellite_bands
        self.n_spectral_indices = n_spectral_indices
        self.n_sensors = n_sensors
        self.n_mask_bits = n_mask_bits
        self.use_mask = use_mask
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # 输入特征维度 = 卫星波段 + 光谱指数 + 传感器独热编码 + 掩码向量（如果使用）
        if use_mask:
            input_dim = n_satellite_bands + n_spectral_indices + n_sensors + n_mask_bits
        else:
            input_dim = n_satellite_bands + n_spectral_indices + n_sensors
        
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
        
        # 输出层：4个无人机波段（G, R, REG, NIR）
        layers.append(nn.Linear(prev_dim, 4))
        # 输出层不使用激活函数，因为反射率值域为[0, 1]或更大，后续可以加Sigmoid如果需要
        
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
        satellite_bands: torch.Tensor,
        spectral_indices: torch.Tensor,
        sensor_onehot: torch.Tensor,
        band_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播。
        
        参数:
            satellite_bands: 卫星波段特征，形状为 (batch_size, n_satellite_bands)
                            统一格式：G, R, REG, NIR（缺失位置填充0）
            spectral_indices: 光谱指数特征，形状为 (batch_size, n_spectral_indices)
            sensor_onehot: 传感器独热编码，形状为 (batch_size, n_sensors)
            band_mask: 波段掩码向量，形状为 (batch_size, n_mask_bits)
                      1表示有效，0表示缺失。如果为None且use_mask=True，则自动生成全1掩码
        
        返回:
            模拟的无人机波段反射率，形状为 (batch_size, 4)，顺序为 [G, R, REG, NIR]
        """
        # 如果使用掩码但未提供，生成全1掩码（所有波段都有效）
        if self.use_mask and band_mask is None:
            batch_size = satellite_bands.shape[0]
            band_mask = torch.ones(batch_size, self.n_mask_bits, device=satellite_bands.device)
        
        # 使用掩码对缺失波段进行屏蔽（将缺失位置的特征置0）
        if self.use_mask and band_mask is not None:
            # 扩展掩码以匹配波段维度
            masked_bands = satellite_bands * band_mask
        else:
            masked_bands = satellite_bands
        
        # 拼接所有输入特征
        if self.use_mask and band_mask is not None:
            x = torch.cat([masked_bands, spectral_indices, sensor_onehot, band_mask], dim=1)
        else:
            x = torch.cat([satellite_bands, spectral_indices, sensor_onehot], dim=1)
        
        # 通过网络
        uav_bands = self.network(x)
        
        return uav_bands
    
    def get_feature_dim(self) -> int:
        """返回输入特征的总维度。"""
        if self.use_mask:
            return self.n_satellite_bands + self.n_spectral_indices + self.n_sensors + self.n_mask_bits
        else:
            return self.n_satellite_bands + self.n_spectral_indices + self.n_sensors

