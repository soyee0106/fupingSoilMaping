"""
完整的两阶段模型（Model C）

将阶段一（差异解耦网络）和阶段二（盐分反演网络）组合在一起，
实现端到端的训练和预测。
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from .stage1_decoder import SensorBiasDecoder
from .stage2_inverter import SalinityInverter
import numpy as np


class FullModelC(nn.Module):
    """
    完整的两阶段模型（Model C）
    
    流程：
    1. 阶段一：卫星特征（波段+指数+传感器标签） -> 模拟无人机波段
    2. 从模拟无人机波段计算光谱指数
    3. 阶段二：模拟无人机波段+指数 -> 盐分值
    """
    
    def __init__(
        self,
        stage1_config: Dict,
        stage2_config: Dict,
        calculate_indices_fn: Optional[callable] = None
    ):
        """
        初始化完整模型。
        
        参数:
            stage1_config: 阶段一网络的配置字典
            stage2_config: 阶段二网络的配置字典
            calculate_indices_fn: 计算光谱指数的函数，如果为None则使用默认函数
        """
        super(FullModelC, self).__init__()
        
        # 创建阶段一和阶段二网络
        self.stage1 = SensorBiasDecoder(**stage1_config)
        self.stage2 = SalinityInverter(**stage2_config)
        
        # 光谱指数计算函数
        if calculate_indices_fn is None:
            self.calculate_indices = self._default_calculate_indices
        else:
            self.calculate_indices = calculate_indices_fn
    
    def _default_calculate_indices(
        self,
        uav_bands: torch.Tensor,
        L: float = 0.5
    ) -> torch.Tensor:
        """
        默认的光谱指数计算函数（在PyTorch中实现），按照S2_indices_calculation.py的逻辑。
        
        参数:
            uav_bands: 无人机波段，形状为 (batch_size, 4)，顺序为 [G, R, REG, NIR]
            L: SAVI土壤调节系数，默认0.5
        
        返回:
            光谱指数，形状为 (batch_size, n_indices)，共26个指数
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
        sensor_onehot: torch.Tensor,
        band_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播。
        
        参数:
            satellite_bands: 卫星波段特征，形状为 (batch_size, n_satellite_bands)
                            统一格式：G, R, REG, NIR（缺失位置填充0）
            satellite_indices: 卫星光谱指数，形状为 (batch_size, n_spectral_indices)
            sensor_onehot: 传感器独热编码，形状为 (batch_size, n_sensors)
            band_mask: 波段掩码向量，形状为 (batch_size, n_mask_bits)
                      1表示有效，0表示缺失
        
        返回:
            包含以下键的字典：
                - 'uav_bands': 模拟的无人机波段，形状为 (batch_size, 4)
                - 'uav_indices': 从模拟波段计算的光谱指数，形状为 (batch_size, n_indices)
                - 'salinity': 预测的盐分值，形状为 (batch_size, 1)
        """
        # 阶段一：卫星特征 -> 模拟无人机波段（传入掩码）
        uav_bands = self.stage1(
            satellite_bands,
            satellite_indices,
            sensor_onehot,
            band_mask
        )
        
        # 从模拟无人机波段计算光谱指数
        uav_indices = self.calculate_indices(uav_bands)
        
        # 阶段二：模拟无人机波段+指数 -> 盐分值
        salinity = self.stage2(uav_bands, uav_indices)
        
        return {
            'uav_bands': uav_bands,
            'uav_indices': uav_indices,
            'salinity': salinity
        }
    
    def predict_salinity(
        self,
        satellite_bands: torch.Tensor,
        satellite_indices: torch.Tensor,
        sensor_onehot: torch.Tensor,
        band_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        仅预测盐分值（便捷方法）。
        
        参数:
            satellite_bands: 卫星波段特征
            satellite_indices: 卫星光谱指数
            sensor_onehot: 传感器独热编码
            band_mask: 波段掩码向量
        
        返回:
            预测的盐分值
        """
        outputs = self.forward(satellite_bands, satellite_indices, sensor_onehot, band_mask)
        return outputs['salinity']

