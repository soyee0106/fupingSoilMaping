"""
光谱指数计算模块

提供各种植被和土壤指数的计算函数，按照S2_indices_calculation.py的逻辑。
支持G, R, REG, NIR四个波段。
"""

import numpy as np
from typing import Dict, Union, Optional


def calculate_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    计算归一化植被指数 (Normalized Difference Vegetation Index, NDVI)
    
    公式: NDVI = (NIR - Red) / (NIR + Red)
    
    参数:
        red: 红光波段反射率
        nir: 近红外波段反射率
    
    返回:
        NDVI值数组，范围通常在[-1, 1]
    """
    denominator = nir + red
    # 避免除零
    denominator = np.where(denominator == 0, np.nan, denominator)
    ndvi = (nir - red) / denominator
    return ndvi


def calculate_si(red: np.ndarray, green: np.ndarray) -> np.ndarray:
    """
    计算盐分指数 (Salinity Index, SI)
    
    公式: SI = sqrt(Red * Green)
    
    参数:
        red: 红光波段反射率
        green: 绿光波段反射率
    
    返回:
        SI值数组
    """
    si = np.sqrt(red * green)
    return si


def calculate_ndsi(
    red: np.ndarray,
    nir: np.ndarray,
    green: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    计算归一化差异盐分指数 (Normalized Difference Salinity Index, NDSI)
    
    公式: NDSI = (Red - NIR) / (Red + NIR)
    或: NDSI = (Green - Red) / (Green + Red)
    
    参数:
        red: 红光波段反射率
        nir: 近红外波段反射率（如果使用第一种公式）
        green: 绿光波段反射率（如果使用第二种公式）
    
    返回:
        NDSI值数组
    """
    if green is not None:
        # 使用绿光和红光
        denominator = green + red
        denominator = np.where(denominator == 0, np.nan, denominator)
        ndsi = (green - red) / denominator
    else:
        # 使用红光和近红外
        denominator = red + nir
        denominator = np.where(denominator == 0, np.nan, denominator)
        ndsi = (red - nir) / denominator
    
    return ndsi


def calculate_evi(
    red: np.ndarray,
    nir: np.ndarray,
    blue: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    计算增强植被指数 (Enhanced Vegetation Index, EVI)
    
    公式: EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
    如果Blue不可用，使用简化公式: EVI = 2.5 * (NIR - Red) / (NIR + 6*Red + 1)
    
    参数:
        red: 红光波段反射率
        nir: 近红外波段反射率
        blue: 蓝光波段反射率（可选）
    
    返回:
        EVI值数组
    """
    if blue is not None:
        denominator = nir + 6 * red - 7.5 * blue + 1
    else:
        denominator = nir + 6 * red + 1
    
    denominator = np.where(denominator == 0, np.nan, denominator)
    evi = 2.5 * (nir - red) / denominator
    return evi


def calculate_savi(
    red: np.ndarray,
    nir: np.ndarray,
    l: float = 0.5
) -> np.ndarray:
    """
    计算土壤调节植被指数 (Soil-Adjusted Vegetation Index, SAVI)
    
    公式: SAVI = (NIR - Red) / (NIR + Red + L) * (1 + L)
    
    参数:
        red: 红光波段反射率
        nir: 近红外波段反射率
        l: 土壤调节因子，通常为0.5
    
    返回:
        SAVI值数组
    """
    denominator = nir + red + l
    denominator = np.where(denominator == 0, np.nan, denominator)
    savi = (nir - red) / denominator * (1 + l)
    return savi


def calculate_ndwi(
    green: np.ndarray,
    nir: np.ndarray
) -> np.ndarray:
    """
    计算归一化差异水分指数 (Normalized Difference Water Index, NDWI)
    
    公式: NDWI = (Green - NIR) / (Green + NIR)
    
    参数:
        green: 绿光波段反射率
        nir: 近红外波段反射率
    
    返回:
        NDWI值数组
    """
    denominator = green + nir
    denominator = np.where(denominator == 0, np.nan, denominator)
    ndwi = (green - nir) / denominator
    return ndwi


def safe_div_array(
    numerator: np.ndarray, denominator: np.ndarray
) -> np.ndarray:
    """数组除法安全处理，避免除以 0。"""
    # 转换为浮点类型，避免类型转换错误
    numerator = numerator.astype(np.float64)
    denominator = denominator.astype(np.float64)
    
    # 避免除以0
    denominator = np.where(denominator == 0, np.nan, denominator)
    
    # 执行除法，结果始终是float64类型
    result = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan, dtype=np.float64), where=denominator != 0)
    return result


def calculate_all_indices(
    bands: Dict[str, np.ndarray],
    index_list: Optional[list] = None,
    L: float = 0.5
) -> Dict[str, np.ndarray]:
    """
    批量计算多个光谱指数，按照S2_indices_calculation.py的逻辑。
    
    参数:
        bands: 波段字典，键为波段名（'G', 'R', 'REG', 'NIR'），值为反射率数组
        index_list: 要计算的指数列表，如果为None则计算所有指数
        L: SAVI土壤调节系数，默认0.5
    
    返回:
        包含所有计算指数的字典
    """
    # 提取波段并转换为浮点类型（避免整数类型导致的计算错误）
    G = bands.get('G', None)
    R = bands.get('R', None)
    REG = bands.get('REG', None)
    NIR = bands.get('NIR', None)
    
    if G is None or R is None or REG is None or NIR is None:
        raise ValueError("All bands (G, R, REG, NIR) must be provided")
    
    # 转换为浮点类型，确保计算精度
    G = np.asarray(G, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    REG = np.asarray(REG, dtype=np.float64)
    NIR = np.asarray(NIR, dtype=np.float64)
    
    indices = {}
    
    # 按照S2_indices_calculation.py的逻辑计算所有指数
    # S1, S1REG
    indices["S1"] = safe_div_array(R * NIR, G)
    indices["S1REG"] = safe_div_array(REG * NIR, G)
    
    # NDSI, NDSIREG
    indices["NDSI"] = safe_div_array(R - NIR, R + NIR)
    indices["NDSIREG"] = safe_div_array(REG - NIR, REG + NIR)
    
    # SI1, SI1REG
    indices["SI1"] = np.sqrt(G * R)
    indices["SI1REG"] = np.sqrt(G * REG)
    
    # SI2, SI2REG
    indices["SI2"] = np.sqrt(G ** 2 + R ** 2 + NIR ** 2)
    indices["SI2REG"] = np.sqrt(G ** 2 + REG ** 2 + NIR ** 2)
    
    # SI3, SI3REG
    indices["SI3"] = np.sqrt(G ** 2 + R ** 2)
    indices["SI3REG"] = np.sqrt(G ** 2 + REG ** 2)
    
    # SIT, SITREG
    indices["SIT"] = 100 * (R - NIR)
    indices["SITREG"] = 100 * (REG - NIR)
    
    # Int1, Int1REG
    indices["Int1"] = (G + R) / 2
    indices["Int1REG"] = (G + REG) / 2
    
    # Int2, Int2REG
    indices["Int2"] = (G + R + NIR) / 2
    indices["Int2REG"] = (G + REG + NIR) / 2
    
    # NDVI, NDVIREG
    indices["NDVI"] = safe_div_array(NIR - R, NIR + R)
    indices["NDVIREG"] = safe_div_array(NIR - REG, NIR + REG)
    
    # SAVI, SAVIREG
    indices["SAVI"] = safe_div_array(
        (1 + L) * (NIR - R), NIR + R + L
    )
    indices["SAVIREG"] = safe_div_array(
        (1 + L) * (NIR - REG), NIR + REG + L
    )
    
    # BI, BIREG
    indices["BI"] = np.sqrt(R ** 2 + NIR ** 2)
    indices["BIREG"] = np.sqrt(REG ** 2 + NIR ** 2)
    
    # IFe2O3, IFe2O3REG
    indices["IFe2O3"] = safe_div_array(R, NIR)
    indices["IFe2O3REG"] = safe_div_array(REG, NIR)
    
    # DVI, DVIREG
    indices["DVI"] = NIR - R
    indices["DVIREG"] = NIR - REG
    
    # 如果指定了index_list，只返回指定的指数
    if index_list is not None:
        filtered_indices = {k: v for k, v in indices.items() if k in index_list}
        return filtered_indices
    
    return indices

