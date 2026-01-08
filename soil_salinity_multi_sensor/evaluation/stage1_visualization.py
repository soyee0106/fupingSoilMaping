"""
Stage 1 可视化实验模块

实验3：光谱曲线可视化
实验4：偏差可视化
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端

from models.stage1_decoder import SensorBiasDecoder
from data_preprocessing.spectral_indices import calculate_all_indices
from data_preprocessing.data_pairing import DataPairer
from data_preprocessing.normalization_utils import load_scalers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_spectrum_at_points(
    raster_path: Path,
    points_gdf: gpd.GeoDataFrame,
    band_indices: List[int],
    scaler: Optional[object] = None
) -> np.ndarray:
    """
    从栅格影像中提取指定点位置的波段值。
    
    参数:
        raster_path: 栅格影像路径
        points_gdf: 点要素GeoDataFrame
        band_indices: 波段索引列表（从1开始）
        scaler: 标准化器（如果提供，会对结果进行反标准化）
    
    返回:
        光谱值数组 (n_points, n_bands)
    """
    with rasterio.open(raster_path) as src:
        # 确保CRS一致
        if points_gdf.crs != src.crs:
            points_gdf = points_gdf.to_crs(src.crs)
        
        spectra = []
        for idx, row in points_gdf.iterrows():
            geom = row.geometry
            point_values = []
            for band_idx in band_indices:
                values = list(src.sample([(geom.x, geom.y)], indexes=[band_idx]))
                if values:
                    point_values.append(values[0][0])
                else:
                    point_values.append(np.nan)
            spectra.append(point_values)
    
    spectra = np.array(spectra, dtype=np.float32)
    
    # 反标准化（如果需要）
    if scaler is not None:
        spectra = scaler.inverse_transform(spectra)
    
    return spectra


def experiment3_spectral_curves_visualization(
    model: SensorBiasDecoder,
    points_shapefile: Path,
    s2_raster_path: Path,
    l8_raster_path: Path,
    uav_raster_path: Path,
    band_mapping_config: Dict,
    scaler_s2: Optional[object] = None,
    scaler_l8: Optional[object] = None,
    scaler_uav: Optional[object] = None,
    output_dir: Path = Path('outputs/stage1_validation/spectral_curves'),
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> None:
    """
    实验3：光谱曲线可视化
    
    选取几个典型地物点，绘制4条曲线：
    1. 真实无人机光谱
    2. 原始S2光谱（重采样后）
    3. 原始L8光谱（重采样后）
    4. 模型预测光谱（S2和L8分别预测）
    
    参数:
        model: Stage 1模型
        points_shapefile: 地类验证点Shapefile路径
        s2_raster_path: S2影像路径（已对齐）
        l8_raster_path: L8影像路径（已对齐）
        uav_raster_path: UAV影像路径（已对齐）
        band_mapping_config: 波段映射配置
        scaler_s2: S2标准化器
        scaler_l8: L8标准化器
        scaler_uav: UAV标准化器
        output_dir: 输出目录
        device: 设备
    """
    logger.info("=" * 60)
    logger.info("实验3：光谱曲线可视化")
    logger.info("=" * 60)
    
    # 读取点要素
    logger.info(f"读取地类验证点: {points_shapefile}")
    points_gdf = gpd.read_file(points_shapefile)
    logger.info(f"共 {len(points_gdf)} 个点")
    
    # 检查是否有地类字段
    landuse_col = None
    for col in points_gdf.columns:
        if col.lower() in ['地类', 'landuse', 'type', 'class', '类别']:
            landuse_col = col
            break
    
    if landuse_col:
        logger.info(f"找到地类字段: {landuse_col}")
        unique_classes = points_gdf[landuse_col].unique()
        logger.info(f"地类: {unique_classes}")
    else:
        logger.warning("未找到地类字段，将绘制所有点")
        landuse_col = None
    
    # 获取波段索引
    pairer = DataPairer()
    # 处理不同的配置结构
    if 's2' in band_mapping_config:
        s2_config = band_mapping_config.get('s2', {}).get('band_mapping', {})
    elif 'satellite' in band_mapping_config and 's2' in band_mapping_config.get('satellite', {}):
        s2_config = band_mapping_config['satellite']['s2'].get('band_mapping', {})
    else:
        s2_config = {}
    
    if 'l8' in band_mapping_config:
        l8_config = band_mapping_config.get('l8', {}).get('band_mapping', {})
    elif 'satellite' in band_mapping_config and 'l8' in band_mapping_config.get('satellite', {}):
        l8_config = band_mapping_config['satellite']['l8'].get('band_mapping', {})
    else:
        l8_config = {}
    
    logger.info(f"S2配置: {s2_config}")
    logger.info(f"L8配置: {l8_config}")
    
    # S2波段索引（G, R, REG, NIR）
    s2_band_indices = []
    for band_name in ['G', 'R', 'REG', 'NIR']:
        if band_name in s2_config:
            indices = s2_config[band_name].get('indices', [])
            if indices:
                s2_band_indices.append(indices[0])
            else:
                s2_band_indices.append(None)
        else:
            s2_band_indices.append(None)
    
    # L8波段索引（G, R, REG, NIR，REG为None）
    l8_band_indices = []
    for band_name in ['G', 'R', 'REG', 'NIR']:
        if band_name == 'REG':
            l8_band_indices.append(None)  # L8没有REG
        elif band_name in l8_config:
            indices = l8_config[band_name].get('indices', [])
            if indices:
                l8_band_indices.append(indices[0])
            else:
                l8_band_indices.append(None)
        else:
            l8_band_indices.append(None)
    
    # UAV波段索引（假设是顺序的4个波段）
    uav_band_indices = [1, 2, 3, 4]
    
    # 提取光谱值
    logger.info("\n提取光谱值...")
    
    # 提取光谱值（直接使用影像中的值，不进行反标准化）
    # 如果影像是标准化后的，那么提取的值就是标准化后的值
    # "原始光谱"在这里指标准化后的值（用于模型输入）
    
    # 提取UAV光谱（真实值，标准化后的）
    logger.info("提取UAV光谱（标准化后的值）...")
    uav_spectra_normalized = extract_spectrum_at_points(
        uav_raster_path, points_gdf, uav_band_indices, None
    )
    logger.info(f"UAV光谱形状: {uav_spectra_normalized.shape}")
    logger.info(f"UAV光谱值范围: [{uav_spectra_normalized.min():.4f}, {uav_spectra_normalized.max():.4f}]")
    
    # 提取S2光谱（标准化后的值）
    # 注意：标准化后的影像只包含4个波段（G, R, REG, NIR），按顺序对应波段索引 1, 2, 3, 4
    # 因此直接使用波段索引 1, 2, 3, 4，而不是原始影像的波段索引
    logger.info("提取S2光谱（标准化后的值）...")
    logger.info(f"S2原始波段索引配置: {s2_band_indices}")
    logger.info(f"S2影像路径: {s2_raster_path}")
    
    # 检查影像波段数
    with rasterio.open(s2_raster_path) as src:
        num_bands = src.count
        logger.info(f"S2影像波段数: {num_bands}")
    
    s2_spectra_normalized = []
    # 标准化后的影像波段顺序：1=G, 2=R, 3=REG, 4=NIR
    normalized_band_order = [1, 2, 3, 4]  # 标准化后影像的波段索引
    band_names = ['G', 'R', 'REG', 'NIR']
    
    for i, (band_name, normalized_idx) in enumerate(zip(band_names, normalized_band_order)):
        if normalized_idx <= num_bands:
            spectra = extract_spectrum_at_points(
                s2_raster_path, points_gdf, [normalized_idx], None
            )
            band_values = spectra[:, 0]
            s2_spectra_normalized.append(band_values)
            logger.info(f"  S2波段 {band_name} (标准化影像索引{normalized_idx}): 有效值 {np.sum(~np.isnan(band_values))}/{len(band_values)}, "
                       f"范围 [{np.nanmin(band_values):.4f}, {np.nanmax(band_values):.4f}]")
        else:
            s2_spectra_normalized.append(np.full(len(points_gdf), np.nan))
            logger.info(f"  S2波段 {band_name}: 波段索引超出范围（填充NaN）")
    s2_spectra_normalized = np.column_stack(s2_spectra_normalized)
    logger.info(f"S2光谱形状: {s2_spectra_normalized.shape}")
    logger.info(f"S2光谱值范围: [{np.nanmin(s2_spectra_normalized):.4f}, {np.nanmax(s2_spectra_normalized):.4f}]")
    logger.info(f"S2光谱有效值数量: {np.sum(~np.isnan(s2_spectra_normalized))}/{s2_spectra_normalized.size}")
    
    # 提取L8光谱（标准化后的值）
    # 注意：标准化后的影像只包含4个波段（G, R, REG=0, NIR），按顺序对应波段索引 1, 2, 3, 4
    # L8没有REG波段，所以索引3位置应该是0或NaN
    logger.info("提取L8光谱（标准化后的值）...")
    logger.info(f"L8原始波段索引配置: {l8_band_indices}")
    logger.info(f"L8影像路径: {l8_raster_path}")
    
    # 检查影像波段数
    with rasterio.open(l8_raster_path) as src:
        num_bands = src.count
        logger.info(f"L8影像波段数: {num_bands}")
    
    l8_spectra_normalized = []
    # 标准化后的影像波段顺序：1=G, 2=R, 3=REG(对L8为0), 4=NIR
    normalized_band_order = [1, 2, 3, 4]  # 标准化后影像的波段索引
    band_names = ['G', 'R', 'REG', 'NIR']
    
    for i, (band_name, normalized_idx) in enumerate(zip(band_names, normalized_band_order)):
        if band_name == 'REG':
            # L8没有REG波段，填充0（标准化后应该是0）
            l8_spectra_normalized.append(np.full(len(points_gdf), 0.0))
            logger.info(f"  L8波段 {band_name}: L8无此波段（填充0）")
        elif normalized_idx <= num_bands:
            spectra = extract_spectrum_at_points(
                l8_raster_path, points_gdf, [normalized_idx], None
            )
            band_values = spectra[:, 0]
            l8_spectra_normalized.append(band_values)
            logger.info(f"  L8波段 {band_name} (标准化影像索引{normalized_idx}): 有效值 {np.sum(~np.isnan(band_values))}/{len(band_values)}, "
                       f"范围 [{np.nanmin(band_values):.4f}, {np.nanmax(band_values):.4f}]")
        else:
            l8_spectra_normalized.append(np.full(len(points_gdf), np.nan))
            logger.info(f"  L8波段 {band_name}: 波段索引超出范围（填充NaN）")
    l8_spectra_normalized = np.column_stack(l8_spectra_normalized)
    logger.info(f"L8光谱形状: {l8_spectra_normalized.shape}")
    logger.info(f"L8光谱值范围: [{np.nanmin(l8_spectra_normalized):.4f}, {np.nanmax(l8_spectra_normalized):.4f}]")
    logger.info(f"L8光谱有效值数量: {np.sum(~np.isnan(l8_spectra_normalized))}/{l8_spectra_normalized.size}")
    
    # 模型预测
    logger.info("模型预测...")
    s2_preds = []
    l8_preds = []
    
    for i in range(len(points_gdf)):
        # S2预测
        # 直接使用标准化后的值（影像中已经是标准化后的）
        s2_bands = s2_spectra_normalized[i].astype(np.float32)
        # 处理NaN值
        s2_bands = np.nan_to_num(s2_bands, nan=0.0)
        
        # 计算光谱指数（需要反标准化后的值来计算指数）
        # 但为了可视化一致性，我们使用标准化后的值计算指数（虽然不太准确，但用于可视化）
        # 或者我们可以跳过指数计算，只使用波段
        # 这里我们使用反标准化后的值计算指数，但可视化时使用标准化后的值
        if scaler_s2 is not None:
            s2_bands_denorm = scaler_s2.inverse_transform(s2_bands.reshape(1, -1)).flatten()
            s2_bands_denorm = np.maximum(s2_bands_denorm, 0.0)  # 确保非负
        else:
            s2_bands_denorm = s2_bands
        
        band_dict = {'G': s2_bands_denorm[0], 'R': s2_bands_denorm[1], 'REG': s2_bands_denorm[2], 'NIR': s2_bands_denorm[3]}
        indices_dict = calculate_all_indices(band_dict)
        index_order = [
            'S1', 'S1REG', 'NDSI', 'NDSIREG', 'SI1', 'SI1REG', 'SI2', 'SI2REG',
            'SI3', 'SI3REG', 'SIT', 'SITREG', 'Int1', 'Int1REG', 'Int2', 'Int2REG',
            'NDVI', 'NDVIREG', 'SAVI', 'SAVIREG', 'BI', 'BIREG', 'IFe2O3', 'IFe2O3REG',
            'DVI', 'DVIREG'
        ]
        s2_indices = np.array([indices_dict[idx] for idx in index_order], dtype=np.float32)
        s2_indices = np.nan_to_num(s2_indices, nan=0.0, posinf=0.0, neginf=0.0)
        
        s2_bands_t = torch.FloatTensor(s2_bands).unsqueeze(0).to(device)
        s2_indices_t = torch.FloatTensor(s2_indices).unsqueeze(0).to(device)
        s2_sensor_t = torch.FloatTensor([[1.0, 0.0]]).to(device)
        s2_mask_t = torch.FloatTensor([[1.0, 1.0, 1.0, 1.0]]).to(device)
        
        with torch.no_grad():
            s2_pred_normalized = model(s2_bands_t, s2_indices_t, s2_sensor_t, s2_mask_t).cpu().numpy().flatten()
        
        # 预测值是标准化后的，直接使用（不反标准化）
        s2_preds.append(s2_pred_normalized)
        
        # L8预测
        # 直接使用标准化后的值
        l8_bands = l8_spectra_normalized[i].astype(np.float32)
        # 处理NaN值
        l8_bands = np.nan_to_num(l8_bands, nan=0.0)
        
        # 计算光谱指数
        if scaler_l8 is not None:
            l8_bands_denorm = scaler_l8.inverse_transform(l8_bands.reshape(1, -1)).flatten()
            l8_bands_denorm = np.maximum(l8_bands_denorm, 0.0)  # 确保非负
        else:
            l8_bands_denorm = l8_bands
        
        band_dict = {'G': l8_bands_denorm[0], 'R': l8_bands_denorm[1], 'REG': l8_bands_denorm[2], 'NIR': l8_bands_denorm[3]}
        indices_dict = calculate_all_indices(band_dict)
        l8_indices = np.array([indices_dict[idx] for idx in index_order], dtype=np.float32)
        l8_indices = np.nan_to_num(l8_indices, nan=0.0, posinf=0.0, neginf=0.0)
        
        l8_bands_t = torch.FloatTensor(l8_bands).unsqueeze(0).to(device)
        l8_indices_t = torch.FloatTensor(l8_indices).unsqueeze(0).to(device)
        l8_sensor_t = torch.FloatTensor([[0.0, 1.0]]).to(device)
        l8_mask_t = torch.FloatTensor([[1.0, 1.0, 0.0, 1.0]]).to(device)
        
        with torch.no_grad():
            l8_pred_normalized = model(l8_bands_t, l8_indices_t, l8_sensor_t, l8_mask_t).cpu().numpy().flatten()
        
        # 预测值是标准化后的，直接使用（不反标准化）
        l8_preds.append(l8_pred_normalized)
    
    s2_preds = np.array(s2_preds)
    l8_preds = np.array(l8_preds)
    
    # 所有光谱值都使用标准化后的值进行可视化
    # "原始光谱"指标准化后的值（用于模型输入）
    # 预测值也是标准化后的，所以所有值都在同一尺度
    uav_spectra_for_plot = uav_spectra_normalized
    s2_spectra_for_plot = s2_spectra_normalized
    l8_spectra_for_plot = l8_spectra_normalized
    
    # 预测值也是标准化后的（模型输出），不需要反标准化
    # 但为了与真实UAV光谱比较，我们需要确保它们在同一尺度
    # 由于模型输出已经是标准化后的，我们直接使用
    s2_preds_normalized = s2_preds.copy()
    l8_preds_normalized = l8_preds.copy()
    
    # 如果预测值被反标准化了，需要重新标准化回来
    # 检查预测值的范围来判断
    if scaler_uav is not None:
        # 如果预测值看起来是反标准化后的（值很大），需要重新标准化
        if np.abs(s2_preds).max() > 10:
            # 预测值已经被反标准化，需要重新标准化用于可视化
            s2_preds_normalized = scaler_uav.transform(s2_preds.reshape(-1, 4)).reshape(-1, 4)
            l8_preds_normalized = scaler_uav.transform(l8_preds.reshape(-1, 4)).reshape(-1, 4)
        else:
            # 预测值已经是标准化后的
            s2_preds_normalized = s2_preds
            l8_preds_normalized = l8_preds
    else:
        s2_preds_normalized = s2_preds
        l8_preds_normalized = l8_preds
    
    # 绘制光谱曲线
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    band_names = ['G', 'R', 'REG', 'NIR']
    band_wavelengths = [560, 665, 705, 842]  # 中心波长（nm）
    
    # 如果有地类字段，按地类分组绘制
    if landuse_col:
        unique_classes = points_gdf[landuse_col].unique()
        for landuse in unique_classes:
            mask = points_gdf[landuse_col] == landuse
            if mask.sum() == 0:
                continue
            
            # 选择几个代表性点（最多5个）
            selected_indices = np.where(mask)[0][:5]
            
            fig, axes = plt.subplots(1, len(selected_indices), figsize=(5*len(selected_indices), 5))
            if len(selected_indices) == 1:
                axes = [axes]
            
            for ax_idx, point_idx in enumerate(selected_indices):
                ax = axes[ax_idx]
                
                # 绘制5条曲线（所有值都是标准化后的，在同一尺度）
                # 处理NaN值：只绘制有效值
                uav_vals = uav_spectra_for_plot[point_idx]
                s2_vals = s2_spectra_for_plot[point_idx]
                l8_vals = l8_spectra_for_plot[point_idx]
                s2_pred_vals = s2_preds_normalized[point_idx]
                l8_pred_vals = l8_preds_normalized[point_idx]
                
                # 创建有效值掩码
                uav_valid = ~np.isnan(uav_vals)
                s2_valid = ~np.isnan(s2_vals)
                l8_valid = ~np.isnan(l8_vals)
                s2_pred_valid = ~np.isnan(s2_pred_vals)
                l8_pred_valid = ~np.isnan(l8_pred_vals)
                
                if np.any(uav_valid):
                    ax.plot(np.array(band_wavelengths)[uav_valid], uav_vals[uav_valid], 'o-', label='真实UAV光谱（标准化后）', linewidth=2, markersize=8, color='blue')
                if np.any(s2_valid):
                    ax.plot(np.array(band_wavelengths)[s2_valid], s2_vals[s2_valid], 's-', label='原始S2光谱（标准化后）', linewidth=2, markersize=8, alpha=0.7, color='orange')
                if np.any(l8_valid):
                    ax.plot(np.array(band_wavelengths)[l8_valid], l8_vals[l8_valid], '^-', label='原始L8光谱（标准化后）', linewidth=2, markersize=8, alpha=0.7, color='green')
                if np.any(s2_pred_valid):
                    ax.plot(np.array(band_wavelengths)[s2_pred_valid], s2_pred_vals[s2_pred_valid], '--', label='S2预测光谱（标准化后）', linewidth=2, alpha=0.8, color='purple')
                if np.any(l8_pred_valid):
                    ax.plot(np.array(band_wavelengths)[l8_pred_valid], l8_pred_vals[l8_pred_valid], '--', label='L8预测光谱（标准化后）', linewidth=2, alpha=0.8, color='gray')
                
                ax.set_xlabel('波长 (nm)', fontsize=12)
                ax.set_ylabel('标准化后的反射率', fontsize=12)
                ax.set_title(f'{landuse} - 点 {point_idx+1}', fontsize=12)
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            output_path = output_dir / f'spectral_curves_{landuse}.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"已保存光谱曲线图: {output_path}")
    else:
        # 绘制所有点（最多10个）
        n_points = min(len(points_gdf), 10)
        
        # 动态计算子图布局：根据实际点数调整
        if n_points <= 5:
            n_rows, n_cols = 1, n_points
            figsize = (5 * n_points, 5)
        else:
            n_rows, n_cols = 2, 5
            figsize = (25, 10)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_points == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_rows > 1 else axes
        
        for i in range(n_points):
            ax = axes[i]
            
            # 绘制5条曲线（所有值都是标准化后的，在同一尺度）
            # 处理NaN值：只绘制有效值
            uav_vals = uav_spectra_for_plot[i]
            s2_vals = s2_spectra_for_plot[i]
            l8_vals = l8_spectra_for_plot[i]
            s2_pred_vals = s2_preds_normalized[i]
            l8_pred_vals = l8_preds_normalized[i]
            
            # 创建有效值掩码
            uav_valid = ~np.isnan(uav_vals)
            s2_valid = ~np.isnan(s2_vals)
            l8_valid = ~np.isnan(l8_vals)
            s2_pred_valid = ~np.isnan(s2_pred_vals)
            l8_pred_valid = ~np.isnan(l8_pred_vals)
            
            if np.any(uav_valid):
                ax.plot(np.array(band_wavelengths)[uav_valid], uav_vals[uav_valid], 'o-', label='真实UAV光谱（标准化后）', linewidth=2, markersize=8, color='blue')
            if np.any(s2_valid):
                ax.plot(np.array(band_wavelengths)[s2_valid], s2_vals[s2_valid], 's-', label='原始S2光谱（标准化后）', linewidth=2, markersize=8, alpha=0.7, color='orange')
            if np.any(l8_valid):
                ax.plot(np.array(band_wavelengths)[l8_valid], l8_vals[l8_valid], '^-', label='原始L8光谱（标准化后）', linewidth=2, markersize=8, alpha=0.7, color='green')
            if np.any(s2_pred_valid):
                ax.plot(np.array(band_wavelengths)[s2_pred_valid], s2_pred_vals[s2_pred_valid], '--', label='S2预测光谱（标准化后）', linewidth=2, alpha=0.8, color='purple')
            if np.any(l8_pred_valid):
                ax.plot(np.array(band_wavelengths)[l8_pred_valid], l8_pred_vals[l8_pred_valid], '--', label='L8预测光谱（标准化后）', linewidth=2, alpha=0.8, color='gray')
            
            ax.set_xlabel('波长 (nm)', fontsize=10)
            ax.set_ylabel('标准化后的反射率', fontsize=10)
            ax.set_title(f'点 {i+1}', fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的空白子图
        if n_points < len(axes):
            for i in range(n_points, len(axes)):
                axes[i].set_visible(False)
        
        plt.tight_layout()
        output_path = output_dir / 'spectral_curves_all.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"已保存光谱曲线图: {output_path}")
    
    logger.info("实验3完成！")


def extract_intermediate_activations(
    model: SensorBiasDecoder,
    satellite_bands: torch.Tensor,
    spectral_indices: torch.Tensor,
    sensor_onehot: torch.Tensor,
    band_mask: torch.Tensor,
    layer_idx: int = 0
) -> torch.Tensor:
    """
    提取模型中间层的激活值。
    
    参数:
        model: Stage 1模型
        satellite_bands: 卫星波段
        spectral_indices: 光谱指数
        sensor_onehot: 传感器独热编码
        band_mask: 波段掩码
        layer_idx: 要提取的层索引（0=第一层Linear+ReLU后，1=第二层Linear+ReLU后，...）
    
    返回:
        激活值张量（ReLU激活后的值）
    """
    # 使用掩码
    if model.use_mask and band_mask is not None:
        masked_bands = satellite_bands * band_mask
    else:
        masked_bands = satellite_bands
    
    # 拼接输入
    if model.use_mask and band_mask is not None:
        x = torch.cat([masked_bands, spectral_indices, sensor_onehot, band_mask], dim=1)
    else:
        x = torch.cat([satellite_bands, spectral_indices, sensor_onehot], dim=1)
    
    # 前向传播到指定层
    # network结构：每层包含 Linear -> BatchNorm -> ReLU -> Dropout
    # 我们提取ReLU激活后的值
    activations = x
    layer_count = 0
    for i, layer in enumerate(model.network):
        activations = layer(activations)
        # 检查是否是ReLU层（每4个层为一组：Linear, BatchNorm, ReLU, Dropout）
        if isinstance(layer, nn.ReLU):
            if layer_count == layer_idx:
                return activations
            layer_count += 1
    
    # 如果没找到指定层，返回最后一层的激活值
    return activations


def experiment4_bias_visualization(
    model: SensorBiasDecoder,
    val_df: pd.DataFrame,
    scaler_s2: Optional[object] = None,
    scaler_l8: Optional[object] = None,
    output_dir: Path = Path('outputs/stage1_validation/bias_analysis'),
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    n_samples: int = 100
) -> None:
    """
    实验4：偏差可视化
    
    固定模型，反推"偏差"：
    - 计算模型内部某一层在S2模式下的激活值 - 在L8模式下的激活值
    - 将差异映射回波段维度
    
    参数:
        model: Stage 1模型
        val_df: 验证集DataFrame
        scaler_s2: S2标准化器
        scaler_l8: L8标准化器
        output_dir: 输出目录
        device: 设备
        n_samples: 分析的样本数
    """
    logger.info("=" * 60)
    logger.info("实验4：偏差可视化")
    logger.info("=" * 60)
    
    # 提取列名
    sat_band_cols = [col for col in val_df.columns if col.startswith('SAT_band_')]
    sat_band_cols.sort()
    
    # 分离S2和L8数据
    s2_df = val_df[val_df['sensor_id'] == 0].copy().reset_index(drop=True)
    l8_df = val_df[val_df['sensor_id'] == 1].copy().reset_index(drop=True)
    
    min_len = min(len(s2_df), len(l8_df), n_samples)
    logger.info(f"分析 {min_len} 对S2-L8样本")
    
    # 提取样本
    s2_activations_list = []
    l8_activations_list = []
    layer_diffs = []
    
    for i in range(min_len):
        # S2样本
        s2_row = s2_df.iloc[i]
        s2_bands = s2_row[sat_band_cols].values.astype(np.float32)
        
        if scaler_s2 is not None:
            if np.abs(s2_bands).max() > 10:
                s2_bands = scaler_s2.transform(s2_bands.reshape(1, -1)).flatten()
        
        band_dict = {'G': s2_bands[0], 'R': s2_bands[1], 'REG': s2_bands[2], 'NIR': s2_bands[3]}
        indices_dict = calculate_all_indices(band_dict)
        index_order = [
            'S1', 'S1REG', 'NDSI', 'NDSIREG', 'SI1', 'SI1REG', 'SI2', 'SI2REG',
            'SI3', 'SI3REG', 'SIT', 'SITREG', 'Int1', 'Int1REG', 'Int2', 'Int2REG',
            'NDVI', 'NDVIREG', 'SAVI', 'SAVIREG', 'BI', 'BIREG', 'IFe2O3', 'IFe2O3REG',
            'DVI', 'DVIREG'
        ]
        s2_indices = np.array([indices_dict[idx] for idx in index_order], dtype=np.float32)
        s2_indices = np.nan_to_num(s2_indices, nan=0.0, posinf=0.0, neginf=0.0)
        
        s2_bands_t = torch.FloatTensor(s2_bands).unsqueeze(0).to(device)
        s2_indices_t = torch.FloatTensor(s2_indices).unsqueeze(0).to(device)
        s2_sensor_t = torch.FloatTensor([[1.0, 0.0]]).to(device)
        s2_mask_t = torch.FloatTensor([[1.0, 1.0, 1.0, 1.0]]).to(device)
        
        # L8样本
        l8_row = l8_df.iloc[i]
        l8_bands = l8_row[sat_band_cols].values.astype(np.float32)
        
        if scaler_l8 is not None:
            if np.abs(l8_bands).max() > 10:
                l8_bands = scaler_l8.transform(l8_bands.reshape(1, -1)).flatten()
        
        band_dict = {'G': l8_bands[0], 'R': l8_bands[1], 'REG': l8_bands[2], 'NIR': l8_bands[3]}
        indices_dict = calculate_all_indices(band_dict)
        l8_indices = np.array([indices_dict[idx] for idx in index_order], dtype=np.float32)
        l8_indices = np.nan_to_num(l8_indices, nan=0.0, posinf=0.0, neginf=0.0)
        
        l8_bands_t = torch.FloatTensor(l8_bands).unsqueeze(0).to(device)
        l8_indices_t = torch.FloatTensor(l8_indices).unsqueeze(0).to(device)
        l8_sensor_t = torch.FloatTensor([[0.0, 1.0]]).to(device)
        l8_mask_t = torch.FloatTensor([[1.0, 1.0, 0.0, 1.0]]).to(device)
        
        # 提取各层激活值
        with torch.no_grad():
            # 第一层激活值
            s2_act1 = extract_intermediate_activations(
                model, s2_bands_t, s2_indices_t, s2_sensor_t, s2_mask_t, layer_idx=0
            )
            l8_act1 = extract_intermediate_activations(
                model, l8_bands_t, l8_indices_t, l8_sensor_t, l8_mask_t, layer_idx=0
            )
            
            s2_activations_list.append(s2_act1.cpu().numpy())
            l8_activations_list.append(l8_act1.cpu().numpy())
            
            # 计算差异
            diff = (s2_act1 - l8_act1).cpu().numpy()
            layer_diffs.append(diff)
    
    s2_activations = np.array(s2_activations_list)  # (n_samples, hidden_dim)
    l8_activations = np.array(l8_activations_list)
    layer_diffs = np.array(layer_diffs)
    
    # 统计分析
    mean_diff = np.mean(layer_diffs, axis=0)
    std_diff = np.std(layer_diffs, axis=0)
    
    logger.info(f"第一层激活值差异统计:")
    logger.info(f"  平均差异: {mean_diff}")
    logger.info(f"  标准差: {std_diff}")
    
    # 可视化
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 激活值差异分布
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # S2激活值分布
    ax = axes[0, 0]
    ax.hist(s2_activations.flatten(), bins=50, alpha=0.7, label='S2', color='blue')
    ax.set_xlabel('激活值')
    ax.set_ylabel('频数')
    ax.set_title('S2激活值分布')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # L8激活值分布
    ax = axes[0, 1]
    ax.hist(l8_activations.flatten(), bins=50, alpha=0.7, label='L8', color='red')
    ax.set_xlabel('激活值')
    ax.set_ylabel('频数')
    ax.set_title('L8激活值分布')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 差异分布
    ax = axes[1, 0]
    ax.hist(layer_diffs.flatten(), bins=50, alpha=0.7, label='差异', color='green')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=2)
    ax.set_xlabel('激活值差异 (S2 - L8)')
    ax.set_ylabel('频数')
    ax.set_title('激活值差异分布')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 平均差异（按神经元）
    ax = axes[1, 1]
    n_neurons = mean_diff.shape[1]
    neuron_indices = np.arange(n_neurons)
    ax.bar(neuron_indices, mean_diff.flatten(), alpha=0.7, color='purple')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('神经元索引')
    ax.set_ylabel('平均激活值差异')
    ax.set_title('各神经元平均差异')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'bias_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"已保存偏差可视化图: {output_path}")
    
    # 2. 尝试将差异映射回波段维度（通过分析输入特征的贡献）
    # 这里简化处理：分析输入特征（波段）对差异的贡献
    logger.info("\n分析输入特征对偏差的贡献...")
    
    # 获取第一层的权重
    first_layer = model.network[0]  # Linear层
    weights = first_layer.weight.data.cpu().numpy()  # (hidden_dim, input_dim)
    
    # 输入特征维度：4波段 + 26指数 + 2传感器 + 4掩码 = 36
    # 分析前4个特征（波段）的权重差异
    band_weights_s2 = weights[:, :4]  # 波段权重
    band_weights_l8 = weights[:, :4]  # 相同（但实际输入不同）
    
    # 计算权重差异（由于S2和L8使用相同的权重，差异主要来自输入）
    # 这里我们分析S2和L8输入的平均差异
    s2_input_bands = []
    l8_input_bands = []
    
    for i in range(min_len):
        s2_row = s2_df.iloc[i]
        s2_bands = s2_row[sat_band_cols].values.astype(np.float32)
        if scaler_s2 is not None and np.abs(s2_bands).max() > 10:
            s2_bands = scaler_s2.transform(s2_bands.reshape(1, -1)).flatten()
        s2_input_bands.append(s2_bands)
        
        l8_row = l8_df.iloc[i]
        l8_bands = l8_row[sat_band_cols].values.astype(np.float32)
        if scaler_l8 is not None and np.abs(l8_bands).max() > 10:
            l8_bands = scaler_l8.transform(l8_bands.reshape(1, -1)).flatten()
        l8_input_bands.append(l8_bands)
    
    s2_input_bands = np.array(s2_input_bands)
    l8_input_bands = np.array(l8_input_bands)
    input_diff = np.mean(s2_input_bands - l8_input_bands, axis=0)
    
    # 可视化输入差异
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    band_names = ['G', 'R', 'REG', 'NIR']
    x_pos = np.arange(len(band_names))
    ax.bar(x_pos, input_diff, alpha=0.7, color='orange')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(band_names)
    ax.set_ylabel('平均输入差异 (S2 - L8)')
    ax.set_title('各波段输入差异')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'input_band_difference.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"已保存输入波段差异图: {output_path}")
    
    # 保存统计结果
    results = {
        'mean_activation_diff': mean_diff.tolist(),
        'std_activation_diff': std_diff.tolist(),
        'mean_input_band_diff': input_diff.tolist(),
        'n_samples': min_len
    }
    
    import json
    results_path = output_dir / 'bias_analysis_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"已保存偏差分析结果: {results_path}")
    logger.info("实验4完成！")

