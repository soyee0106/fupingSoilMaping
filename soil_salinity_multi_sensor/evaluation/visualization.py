"""
可视化模块

提供各种可视化功能：
- 损失曲线
- 散点图（真实值vs预测值）
- 盐分空间分布图
- 波段偏差柱状图
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_loss_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    save_path: Optional[Path] = None,
    title: str = "Training and Validation Loss",
    figsize: Tuple[int, int] = (10, 6)
):
    """
    绘制损失曲线。
    
    参数:
        train_losses: 训练损失列表
        val_losses: 验证损失列表（可选）
        save_path: 保存路径
        title: 图表标题
        figsize: 图表大小
    """
    plt.figure(figsize=figsize)
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    
    if val_losses is not None:
        plt.plot(val_losses, label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Loss curve saved to {save_path}")
    
    plt.close()


def plot_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Optional[Dict[str, float]] = None,
    save_path: Optional[Path] = None,
    title: str = "Predicted vs Actual Salinity",
    figsize: Tuple[int, int] = (8, 8),
    color: str = 'blue',
    alpha: float = 0.6
):
    """
    绘制散点图（真实值vs预测值）。
    
    参数:
        y_true: 真实值
        y_pred: 预测值
        metrics: 评估指标字典（可选，用于在图上显示）
        save_path: 保存路径
        title: 图表标题
        figsize: 图表大小
        color: 散点颜色
        alpha: 透明度
    """
    plt.figure(figsize=figsize)
    
    # 绘制散点
    plt.scatter(y_true, y_pred, alpha=alpha, color=color, s=50, edgecolors='black', linewidth=0.5)
    
    # 绘制1:1线
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 Line')
    
    # 添加指标文本
    if metrics is not None:
        text_str = ""
        if 'R2' in metrics:
            text_str += f"R² = {metrics['R2']:.4f}\n"
        if 'RMSE' in metrics:
            text_str += f"RMSE = {metrics['RMSE']:.4f}\n"
        if 'MAE' in metrics:
            text_str += f"MAE = {metrics['MAE']:.4f}"
        
        plt.text(0.05, 0.95, text_str, transform=plt.gca().transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.xlabel('Actual Salinity', fontsize=12)
    plt.ylabel('Predicted Salinity', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Scatter plot saved to {save_path}")
    
    plt.close()


def plot_salinity_distribution(
    salinity_values: np.ndarray,
    coordinates: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None,
    title: str = "Salinity Distribution",
    figsize: Tuple[int, int] = (12, 8),
    cmap: str = 'YlOrRd'
):
    """
    绘制盐分空间分布图。
    
    参数:
        salinity_values: 盐分值数组
        coordinates: 坐标数组，形状为 (n_samples, 2)，列分别为x和y
        save_path: 保存路径
        title: 图表标题
        figsize: 图表大小
        cmap: 颜色映射
    """
    if coordinates is not None:
        # 空间分布图
        plt.figure(figsize=figsize)
        scatter = plt.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=salinity_values,
            cmap=cmap,
            s=50,
            edgecolors='black',
            linewidth=0.5,
            alpha=0.8
        )
        plt.colorbar(scatter, label='Salinity')
        plt.xlabel('Longitude', fontsize=12)
        plt.ylabel('Latitude', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
    else:
        # 直方图
        plt.figure(figsize=figsize)
        plt.hist(salinity_values, bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel('Salinity', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Salinity distribution plot saved to {save_path}")
    
    plt.close()


def plot_band_bias(
    true_bands: np.ndarray,
    pred_bands: np.ndarray,
    band_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    title: str = "Band Prediction Bias",
    figsize: Tuple[int, int] = (10, 6)
):
    """
    绘制波段偏差柱状图。
    
    参数:
        true_bands: 真实波段值，形状为 (n_samples, n_bands)
        pred_bands: 预测波段值，形状为 (n_samples, n_bands)
        band_names: 波段名称列表
        save_path: 保存路径
        title: 图表标题
        figsize: 图表大小
    """
    if band_names is None:
        band_names = [f'Band {i+1}' for i in range(true_bands.shape[1])]
    
    # 计算每个波段的平均偏差
    biases = np.mean(pred_bands - true_bands, axis=0)
    stds = np.std(pred_bands - true_bands, axis=0)
    
    plt.figure(figsize=figsize)
    x_pos = np.arange(len(band_names))
    bars = plt.bar(x_pos, biases, yerr=stds, capsize=5, alpha=0.7, edgecolor='black')
    
    # 添加零线
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
    
    plt.xlabel('Band', fontsize=12)
    plt.ylabel('Bias (Predicted - True)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(x_pos, band_names)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Band bias plot saved to {save_path}")
    
    plt.close()


def plot_uncertainty_map(
    uncertainty_values: np.ndarray,
    coordinates: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "Uncertainty Map",
    figsize: Tuple[int, int] = (12, 8),
    cmap: str = 'Reds'
):
    """
    绘制不确定性分布图。
    
    参数:
        uncertainty_values: 不确定性值数组
        coordinates: 坐标数组
        save_path: 保存路径
        title: 图表标题
        figsize: 图表大小
        cmap: 颜色映射
    """
    plt.figure(figsize=figsize)
    scatter = plt.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=uncertainty_values,
        cmap=cmap,
        s=50,
        edgecolors='black',
        linewidth=0.5,
        alpha=0.8
    )
    plt.colorbar(scatter, label='Uncertainty')
    plt.xlabel('Longitude', fontsize=12)
    plt.ylabel('Latitude', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Uncertainty map saved to {save_path}")
    
    plt.close()

