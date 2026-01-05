"""
评估与可视化模块

该模块包含模型评估和结果可视化的功能：
- 回归评价指标计算（R², RMSE, MAE等）
- 不确定性分析（多传感器结果差异）
- 可视化（损失曲线、散点图、空间分布图等）
"""

from .metrics import (
    calculate_r2,
    calculate_rmse,
    calculate_mae,
    calculate_mape,
    calculate_all_metrics
)
from .uncertainty_analysis import (
    calculate_sensor_differences,
    generate_uncertainty_map
)
from .visualization import (
    plot_loss_curves,
    plot_scatter,
    plot_salinity_distribution,
    plot_band_bias
)

__all__ = [
    'calculate_r2',
    'calculate_rmse',
    'calculate_mae',
    'calculate_mape',
    'calculate_all_metrics',
    'calculate_sensor_differences',
    'generate_uncertainty_map',
    'plot_loss_curves',
    'plot_scatter',
    'plot_salinity_distribution',
    'plot_band_bias',
]

