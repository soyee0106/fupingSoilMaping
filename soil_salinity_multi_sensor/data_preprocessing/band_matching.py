"""
波段匹配模块

定义无人机波段与卫星波段（Sentinel-2, Landsat-8）的对应关系。
基于中心波长和带宽进行匹配。
"""

from typing import Dict, List, Tuple


def create_band_mapping() -> Dict[str, Dict[str, List[int]]]:
    """
    创建波段映射字典，定义无人机波段与卫星波段的对应关系。
    
    返回:
        Dict: 包含以下键的字典：
            - 'uav_bands': 无人机波段名称列表（4个波段）
            - 's2_mapping': Sentinel-2波段到无人机波段的映射
            - 'l8_mapping': Landsat-8波段到无人机波段的映射
            - 'band_centers': 各波段的中心波长（nm）
            - 'band_widths': 各波段的带宽（nm）
    
    示例:
        >>> mapping = create_band_mapping()
        >>> print(mapping['uav_bands'])
        ['B', 'G', 'R', 'NIR']
    """
    # 定义无人机4个波段（通常为：蓝、绿、红、近红外）
    uav_bands = ['B', 'G', 'R', 'NIR']
    
    # Sentinel-2波段信息（中心波长，单位：nm）
    # S2波段：B2(490nm), B3(560nm), B4(665nm), B8(842nm)
    s2_band_centers = {
        'B2': 490,   # 蓝光
        'B3': 560,   # 绿光
        'B4': 665,   # 红光
        'B8': 842,   # 近红外
    }
    
    # Landsat-8波段信息（中心波长，单位：nm）
    # L8波段：B2(482nm), B3(561nm), B4(655nm), B5(865nm)
    l8_band_centers = {
        'B2': 482,   # 蓝光
        'B3': 561,   # 绿光
        'B4': 655,   # 红光
        'B5': 865,   # 近红外
    }
    
    # 无人机波段中心波长（假设值，实际应根据具体传感器调整）
    uav_band_centers = {
        'B': 475,    # 蓝光
        'G': 560,    # 绿光
        'R': 650,    # 红光
        'NIR': 840,  # 近红外
    }
    
    # 创建映射关系（基于最接近的中心波长）
    s2_mapping = {
        'B2': 'B',   # S2蓝光 -> UAV蓝光
        'B3': 'G',   # S2绿光 -> UAV绿光
        'B4': 'R',   # S2红光 -> UAV红光
        'B8': 'NIR', # S2近红外 -> UAV近红外
    }
    
    l8_mapping = {
        'B2': 'B',   # L8蓝光 -> UAV蓝光
        'B3': 'G',   # L8绿光 -> UAV绿光
        'B4': 'R',   # L8红光 -> UAV红光
        'B5': 'NIR', # L8近红外 -> UAV近红外
    }
    
    # 波段宽度（nm，用于更精确的匹配）
    band_widths = {
        'UAV': {'B': 50, 'G': 50, 'R': 50, 'NIR': 50},
        'S2': {'B2': 65, 'B3': 35, 'B4': 30, 'B8': 115},
        'L8': {'B2': 60, 'B3': 57, 'B4': 57, 'B5': 28},
    }
    
    return {
        'uav_bands': uav_bands,
        's2_mapping': s2_mapping,
        'l8_mapping': l8_mapping,
        's2_band_centers': s2_band_centers,
        'l8_band_centers': l8_band_centers,
        'uav_band_centers': uav_band_centers,
        'band_widths': band_widths,
    }


def find_closest_band(
    target_wavelength: float,
    available_bands: Dict[str, float],
    max_diff: float = 50.0
) -> Tuple[str, float]:
    """
    根据中心波长找到最接近的波段。
    
    参数:
        target_wavelength: 目标中心波长（nm）
        available_bands: 可用波段字典 {波段名: 中心波长}
        max_diff: 最大允许波长差（nm）
    
    返回:
        Tuple[str, float]: (波段名, 波长差)
    """
    best_band = None
    min_diff = float('inf')
    
    for band_name, wavelength in available_bands.items():
        diff = abs(wavelength - target_wavelength)
        if diff < min_diff:
            min_diff = diff
            best_band = band_name
    
    if min_diff > max_diff:
        return None, min_diff
    
    return best_band, min_diff

