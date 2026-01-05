"""
不确定性分析模块

实现生成多传感器结果差异图（不确定性图）的功能。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_sensor_differences(
    predictions: Dict[str, np.ndarray],
    sensor_names: Optional[List[str]] = None
) -> Dict[str, np.ndarray]:
    """
    计算不同传感器预测结果之间的差异。
    
    参数:
        predictions: 字典，键为传感器名，值为预测值数组
        sensor_names: 传感器名称列表，如果为None则使用predictions的键
    
    返回:
        包含差异统计的字典：
            - 'mean_diff': 平均差异
            - 'std_diff': 标准差
            - 'max_diff': 最大差异
            - 'min_diff': 最小差异
            - 'pairwise_diffs': 两两传感器之间的差异矩阵
    """
    if sensor_names is None:
        sensor_names = list(predictions.keys())
    
    n_sensors = len(sensor_names)
    n_samples = len(predictions[sensor_names[0]])
    
    # 计算所有传感器预测的平均值和标准差
    all_predictions = np.array([predictions[sensor] for sensor in sensor_names])
    mean_pred = np.mean(all_predictions, axis=0)
    std_pred = np.std(all_predictions, axis=0)
    
    # 计算两两传感器之间的差异
    pairwise_diffs = {}
    for i, sensor1 in enumerate(sensor_names):
        for j, sensor2 in enumerate(sensor_names):
            if i < j:
                diff = np.abs(predictions[sensor1] - predictions[sensor2])
                pairwise_diffs[f"{sensor1}_vs_{sensor2}"] = diff
    
    results = {
        'mean_prediction': mean_pred,
        'std_prediction': std_pred,
        'mean_diff': np.mean(std_pred),
        'std_diff': np.std(std_pred),
        'max_diff': np.max(std_pred),
        'min_diff': np.min(std_pred),
        'pairwise_diffs': pairwise_diffs,
        'sensor_names': sensor_names
    }
    
    logger.info(f"Calculated differences for {n_sensors} sensors")
    logger.info(f"Mean difference: {results['mean_diff']:.4f}")
    logger.info(f"Std difference: {results['std_diff']:.4f}")
    
    return results


def generate_uncertainty_map(
    predictions: Dict[str, np.ndarray],
    coordinates: Optional[np.ndarray] = None,
    method: str = 'std'
) -> np.ndarray:
    """
    生成不确定性图（基于多传感器预测的标准差或范围）。
    
    参数:
        predictions: 字典，键为传感器名，值为预测值数组
        coordinates: 坐标数组（可选，用于空间可视化）
        method: 不确定性计算方法，'std'（标准差）或'range'（范围）
    
    返回:
        不确定性值数组
    """
    sensor_names = list(predictions.keys())
    n_samples = len(predictions[sensor_names[0]])
    
    # 将所有预测值堆叠
    all_predictions = np.array([predictions[sensor] for sensor in sensor_names])
    
    if method == 'std':
        # 使用标准差作为不确定性
        uncertainty = np.std(all_predictions, axis=0)
    elif method == 'range':
        # 使用范围（最大值-最小值）作为不确定性
        uncertainty = np.max(all_predictions, axis=0) - np.min(all_predictions, axis=0)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'std' or 'range'.")
    
    logger.info(f"Generated uncertainty map using method: {method}")
    logger.info(f"Uncertainty range: [{np.min(uncertainty):.4f}, {np.max(uncertainty):.4f}]")
    
    return uncertainty


def calculate_uncertainty_metrics(
    uncertainty_values: np.ndarray
) -> Dict[str, float]:
    """
    计算不确定性统计指标。
    
    参数:
        uncertainty_values: 不确定性值数组
    
    返回:
        包含统计指标的字典
    """
    metrics = {
        'mean': np.mean(uncertainty_values),
        'std': np.std(uncertainty_values),
        'median': np.median(uncertainty_values),
        'min': np.min(uncertainty_values),
        'max': np.max(uncertainty_values),
        'q25': np.percentile(uncertainty_values, 25),
        'q75': np.percentile(uncertainty_values, 75),
    }
    
    return metrics

