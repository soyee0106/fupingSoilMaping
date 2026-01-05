"""
基线模型定义

包含两个基线模型用于对比：
- Model A: 直接从卫星特征预测盐分（单阶段）
- Model B: 卫星特征 -> 无人机波段 -> 盐分（两阶段，但阶段一不学习传感器偏差）
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from .stage1_decoder import SensorBiasDecoder
from .stage2_inverter import SalinityInverter


class BaselineModelA(nn.Module):
    """
    基线模型A：直接从卫星特征预测盐分
    
    输入：卫星波段 + 光谱指数 + 传感器标签
    输出：盐分值
    
    这是一个单阶段的端到端模型，不进行光谱解耦。
    """
    
    def __init__(
        self,
        n_satellite_bands: int = 4,
        n_spectral_indices: int = 6,
        n_sensors: int = 2,
        hidden_dims: list = [128, 64, 32, 16],
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True
    ):
        """
        初始化基线模型A。
        
        参数:
            n_satellite_bands: 卫星波段数量
            n_spectral_indices: 光谱指数数量
            n_sensors: 传感器数量
            hidden_dims: 隐藏层维度列表
            dropout_rate: Dropout比率
            use_batch_norm: 是否使用批归一化
        """
        super(BaselineModelA, self).__init__()
        
        # 输入特征维度
        input_dim = n_satellite_bands + n_spectral_indices + n_sensors
        
        # 构建全连接层
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # 输出层：盐分值
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
        satellite_bands: torch.Tensor,
        spectral_indices: torch.Tensor,
        sensor_onehot: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播。
        
        参数:
            satellite_bands: 卫星波段特征
            spectral_indices: 光谱指数特征
            sensor_onehot: 传感器独热编码
        
        返回:
            预测的盐分值，形状为 (batch_size, 1)
        """
        # 拼接所有输入特征
        x = torch.cat([satellite_bands, spectral_indices, sensor_onehot], dim=1)
        
        # 通过网络
        salinity = self.network(x)
        
        return salinity


class BaselineModelB(nn.Module):
    """
    基线模型B：卫星特征 -> 无人机波段 -> 盐分
    
    这是一个两阶段模型，但阶段一不学习传感器偏差（不使用传感器标签）。
    阶段一：卫星波段+指数 -> 无人机波段
    阶段二：无人机波段+指数 -> 盐分
    """
    
    def __init__(
        self,
        stage1_config: Dict,
        stage2_config: Dict,
        calculate_indices_fn: Optional[callable] = None
    ):
        """
        初始化基线模型B。
        
        参数:
            stage1_config: 阶段一网络配置（注意：不使用传感器标签）
            stage2_config: 阶段二网络配置
            calculate_indices_fn: 计算光谱指数的函数
        """
        super(BaselineModelB, self).__init__()
        
        # 阶段一：不使用传感器标签的版本
        # 修改输入维度（减去传感器维度）
        stage1_config_no_sensor = stage1_config.copy()
        n_sensors = stage1_config_no_sensor.pop('n_sensors', 2)
        n_satellite_bands = stage1_config_no_sensor.get('n_satellite_bands', 4)
        n_spectral_indices = stage1_config_no_sensor.get('n_spectral_indices', 6)
        
        # 创建阶段一网络（不使用传感器标签）
        from .stage1_decoder import SensorBiasDecoder
        
        # 临时修改输入维度
        original_input_dim = n_satellite_bands + n_spectral_indices + n_sensors
        new_input_dim = n_satellite_bands + n_spectral_indices
        
        # 构建阶段一网络（不使用传感器标签）
        self.stage1 = self._create_stage1_no_sensor(
            n_satellite_bands=n_satellite_bands,
            n_spectral_indices=n_spectral_indices,
            hidden_dims=stage1_config_no_sensor.get('hidden_dims', [128, 64, 32]),
            dropout_rate=stage1_config_no_sensor.get('dropout_rate', 0.3),
            use_batch_norm=stage1_config_no_sensor.get('use_batch_norm', True)
        )
        
        # 创建阶段二网络
        from .stage2_inverter import SalinityInverter
        self.stage2 = SalinityInverter(**stage2_config)
        
        # 光谱指数计算函数
        if calculate_indices_fn is None:
            self.calculate_indices = self._default_calculate_indices
        else:
            self.calculate_indices = calculate_indices_fn
    
    def _create_stage1_no_sensor(
        self,
        n_satellite_bands: int,
        n_spectral_indices: int,
        hidden_dims: list,
        dropout_rate: float,
        use_batch_norm: bool
    ) -> nn.Module:
        """创建不使用传感器标签的阶段一网络。"""
        input_dim = n_satellite_bands + n_spectral_indices
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 4))
        
        network = nn.Sequential(*layers)
        
        # 初始化权重
        for m in network.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        return network
    
    def _default_calculate_indices(self, uav_bands: torch.Tensor, L: float = 0.5) -> torch.Tensor:
        """
        默认的光谱指数计算函数，按照S2_indices_calculation.py的逻辑。
        
        参数:
            uav_bands: 无人机波段，形状为 (batch_size, 4)，顺序为 [G, R, REG, NIR]
            L: SAVI土壤调节系数，默认0.5
        
        返回:
            光谱指数，形状为 (batch_size, 26)
        """
        batch_size = uav_bands.shape[0]
        device = uav_bands.device
        
        # 提取各波段（顺序：G, R, REG, NIR）
        G = uav_bands[:, 0]
        R = uav_bands[:, 1]
        REG = uav_bands[:, 2]
        NIR = uav_bands[:, 3]
        
        # 安全除法函数
        def safe_div_torch(num, den):
            den = torch.where(den == 0, torch.tensor(1e-8, device=device), den)
            return num / den
        
        indices_list = []
        
        # S1, S1REG
        indices_list.append(safe_div_torch(R * NIR, G).unsqueeze(1))  # S1
        indices_list.append(safe_div_torch(REG * NIR, G).unsqueeze(1))  # S1REG
        
        # NDSI, NDSIREG
        indices_list.append(safe_div_torch(R - NIR, R + NIR).unsqueeze(1))  # NDSI
        indices_list.append(safe_div_torch(REG - NIR, REG + NIR).unsqueeze(1))  # NDSIREG
        
        # SI1, SI1REG
        indices_list.append(torch.sqrt(G * R).unsqueeze(1))  # SI1
        indices_list.append(torch.sqrt(G * REG).unsqueeze(1))  # SI1REG
        
        # SI2, SI2REG
        indices_list.append(torch.sqrt(G ** 2 + R ** 2 + NIR ** 2).unsqueeze(1))  # SI2
        indices_list.append(torch.sqrt(G ** 2 + REG ** 2 + NIR ** 2).unsqueeze(1))  # SI2REG
        
        # SI3, SI3REG
        indices_list.append(torch.sqrt(G ** 2 + R ** 2).unsqueeze(1))  # SI3
        indices_list.append(torch.sqrt(G ** 2 + REG ** 2).unsqueeze(1))  # SI3REG
        
        # SIT, SITREG
        indices_list.append((100 * (R - NIR)).unsqueeze(1))  # SIT
        indices_list.append((100 * (REG - NIR)).unsqueeze(1))  # SITREG
        
        # Int1, Int1REG
        indices_list.append(((G + R) / 2).unsqueeze(1))  # Int1
        indices_list.append(((G + REG) / 2).unsqueeze(1))  # Int1REG
        
        # Int2, Int2REG
        indices_list.append(((G + R + NIR) / 2).unsqueeze(1))  # Int2
        indices_list.append(((G + REG + NIR) / 2).unsqueeze(1))  # Int2REG
        
        # NDVI, NDVIREG
        indices_list.append(safe_div_torch(NIR - R, NIR + R).unsqueeze(1))  # NDVI
        indices_list.append(safe_div_torch(NIR - REG, NIR + REG).unsqueeze(1))  # NDVIREG
        
        # SAVI, SAVIREG
        indices_list.append(safe_div_torch((1 + L) * (NIR - R), NIR + R + L).unsqueeze(1))  # SAVI
        indices_list.append(safe_div_torch((1 + L) * (NIR - REG), NIR + REG + L).unsqueeze(1))  # SAVIREG
        
        # BI, BIREG
        indices_list.append(torch.sqrt(R ** 2 + NIR ** 2).unsqueeze(1))  # BI
        indices_list.append(torch.sqrt(REG ** 2 + NIR ** 2).unsqueeze(1))  # BIREG
        
        # IFe2O3, IFe2O3REG
        indices_list.append(safe_div_torch(R, NIR).unsqueeze(1))  # IFe2O3
        indices_list.append(safe_div_torch(REG, NIR).unsqueeze(1))  # IFe2O3REG
        
        # DVI, DVIREG
        indices_list.append((NIR - R).unsqueeze(1))  # DVI
        indices_list.append((NIR - REG).unsqueeze(1))  # DVIREG
        
        # 拼接所有指数（共26个）
        indices = torch.cat(indices_list, dim=1)
        return indices
    
    def forward(
        self,
        satellite_bands: torch.Tensor,
        satellite_indices: torch.Tensor,
        sensor_onehot: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播（不使用传感器标签）。
        
        参数:
            satellite_bands: 卫星波段特征
            satellite_indices: 卫星光谱指数
            sensor_onehot: 传感器独热编码（不使用，但保留接口兼容性）
        
        返回:
            包含以下键的字典：
                - 'uav_bands': 模拟的无人机波段
                - 'uav_indices': 从模拟波段计算的光谱指数
                - 'salinity': 预测的盐分值
        """
        # 阶段一：卫星特征 -> 模拟无人机波段（不使用传感器标签）
        x_stage1 = torch.cat([satellite_bands, satellite_indices], dim=1)
        uav_bands = self.stage1(x_stage1)
        
        # 从模拟无人机波段计算光谱指数
        uav_indices = self.calculate_indices(uav_bands)
        
        # 阶段二：模拟无人机波段+指数 -> 盐分值
        salinity = self.stage2(uav_bands, uav_indices)
        
        return {
            'uav_bands': uav_bands,
            'uav_indices': uav_indices,
            'salinity': salinity
        }

