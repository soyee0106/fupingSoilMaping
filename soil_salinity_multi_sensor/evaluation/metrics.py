"""
评估指标模块

实现常见的回归评价指标，如R²、RMSE、MAE等。
"""

import numpy as np
from typing import Dict, Union
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error
)


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算决定系数（R²）。
    
    参数:
        y_true: 真实值
        y_pred: 预测值
    
    返回:
        R²值
    """
    return r2_score(y_true, y_pred)


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算均方根误差（RMSE）。
    
    参数:
        y_true: 真实值
        y_pred: 预测值
    
    返回:
        RMSE值
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算平均绝对误差（MAE）。
    
    参数:
        y_true: 真实值
        y_pred: 预测值
    
    返回:
        MAE值
    """
    return mean_absolute_error(y_true, y_pred)


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算平均绝对百分比误差（MAPE）。
    
    参数:
        y_true: 真实值
        y_pred: 预测值
    
    返回:
        MAPE值（百分比）
    """
    # 避免除零
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    
    return mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100


def calculate_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算均方误差（MSE）。
    
    参数:
        y_true: 真实值
        y_pred: 预测值
    
    返回:
        MSE值
    """
    return mean_squared_error(y_true, y_pred)


def calculate_median_ae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算中位数绝对误差（Median AE）。
    
    参数:
        y_true: 真实值
        y_pred: 预测值
    
    返回:
        中位数绝对误差
    """
    return np.median(np.abs(y_true - y_pred))


def calculate_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算皮尔逊相关系数。
    
    参数:
        y_true: 真实值
        y_pred: 预测值
    
    返回:
        相关系数
    """
    return np.corrcoef(y_true, y_pred)[0, 1]


def calculate_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics_list: list = None
) -> Dict[str, float]:
    """
    计算所有评估指标。
    
    参数:
        y_true: 真实值
        y_pred: 预测值
        metrics_list: 要计算的指标列表，如果为None则计算所有指标
    
    返回:
        包含所有指标值的字典
    """
    if metrics_list is None:
        metrics_list = [
            'R2', 'RMSE', 'MAE', 'MSE', 'MAPE',
            'Median_AE', 'Correlation'
        ]
    
    metrics = {}
    
    if 'R2' in metrics_list:
        metrics['R2'] = calculate_r2(y_true, y_pred)
    
    if 'RMSE' in metrics_list:
        metrics['RMSE'] = calculate_rmse(y_true, y_pred)
    
    if 'MAE' in metrics_list:
        metrics['MAE'] = calculate_mae(y_true, y_pred)
    
    if 'MSE' in metrics_list:
        metrics['MSE'] = calculate_mse(y_true, y_pred)
    
    if 'MAPE' in metrics_list:
        metrics['MAPE'] = calculate_mape(y_true, y_pred)
    
    if 'Median_AE' in metrics_list:
        metrics['Median_AE'] = calculate_median_ae(y_true, y_pred)
    
    if 'Correlation' in metrics_list:
        metrics['Correlation'] = calculate_correlation(y_true, y_pred)
    
    return metrics

