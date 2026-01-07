"""
影像对齐和重采样模块

实现卫星影像和无人机影像的空间对齐、重采样到一致分辨率，
并提取对齐区域内的所有像元对，用于学习密集映射关系。
"""

import numpy as np
import rasterio
from rasterio import warp
from rasterio.transform import Affine
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def aggregate_to_target_resolution(
    high_res_array: np.ndarray,
    high_res_transform: Affine,
    target_resolution: float,
    nodata_value: Optional[float] = None
) -> Tuple[np.ndarray, Affine]:
    """
    将高分辨率影像聚合成目标分辨率。
    
    参数:
        high_res_array: 高分辨率影像数组
        high_res_transform: 高分辨率影像的transform
        target_resolution: 目标分辨率（度或米，取决于CRS）
        nodata_value: 无效值，聚合时会被排除
    
    返回:
        聚合后的影像数组和新的transform
    """
    # 计算当前分辨率
    current_res_x = abs(high_res_transform[0])
    current_res_y = abs(high_res_transform[4])
    
    # 计算聚合比例
    factor_x = target_resolution / current_res_x
    factor_y = target_resolution / current_res_y
    
    if factor_x < 1.0 or factor_y < 1.0:
        raise ValueError(f"目标分辨率 ({target_resolution}) 不能小于当前分辨率 ({current_res_x}, {current_res_y})")
    
    factor_x_int = int(factor_x)
    factor_y_int = int(factor_y)
    
    h, w = high_res_array.shape
    
    # 裁剪到能被聚合比例整除的尺寸
    h_new = (h // factor_y_int) * factor_y_int
    w_new = (w // factor_x_int) * factor_x_int
    array_cropped = high_res_array[:h_new, :w_new]
    
    # 处理nodata值
    if nodata_value is not None:
        array_masked = array_cropped.astype(np.float64)
        array_masked[array_cropped == nodata_value] = np.nan
    else:
        array_masked = array_cropped.astype(np.float64)
    
    # 块聚合：使用reshape + nanmean
    h_out = h_new // factor_y_int
    w_out = w_new // factor_x_int
    
    # Reshape为块结构并求均值
    reshaped = array_masked.reshape(h_out, factor_y_int, w_out, factor_x_int)
    aggregated = np.nanmean(reshaped, axis=(1, 3))
    
    # 更新transform
    new_transform = Affine(
        high_res_transform[0] * factor_x_int,
        high_res_transform[1],
        high_res_transform[2],
        high_res_transform[3],
        high_res_transform[4] * factor_y_int,
        high_res_transform[5]
    )
    
    return aggregated, new_transform


def align_images_to_grid(
    source_bands: List[np.ndarray],
    source_transform: Affine,
    source_crs: rasterio.crs.CRS,
    target_raster_path: Path,
    resampling_method: warp.Resampling = warp.Resampling.nearest
) -> Tuple[List[np.ndarray], Affine, Tuple[int, int]]:
    """
    将源影像对齐到目标影像的像元网格。
    
    参数:
        source_bands: 源影像波段列表
        source_transform: 源影像的transform
        source_crs: 源影像的CRS
        target_raster_path: 目标影像路径（用于获取网格信息）
        resampling_method: 重采样方法
    
    返回:
        对齐后的波段列表、新的transform、输出尺寸
    """
    # 读取目标影像信息
    with rasterio.open(target_raster_path) as target_src:
        target_transform = target_src.transform
        target_crs = target_src.crs
        target_height = target_src.height
        target_width = target_src.width
        target_bounds = target_src.bounds
    
    logger.info(f"Target raster: {target_raster_path}")
    logger.info(f"  Size: {target_height} × {target_width}")
    logger.info(f"  CRS: {target_crs}")
    logger.info(f"  Bounds: {target_bounds}")
    
    # 计算重叠区域
    source_height, source_width = source_bands[0].shape
    source_bounds = rasterio.transform.array_bounds(source_height, source_width, source_transform)
    # array_bounds 返回 (left, bottom, right, top) tuple
    source_left, source_bottom, source_right, source_top = source_bounds
    
    # 计算重叠bounds
    overlap_left = max(source_left, target_bounds.left)
    overlap_right = min(source_right, target_bounds.right)
    overlap_bottom = max(source_bottom, target_bounds.bottom)
    overlap_top = min(source_top, target_bounds.top)
    
    if overlap_left >= overlap_right or overlap_bottom >= overlap_top:
        raise ValueError("源影像和目标影像没有重叠区域")
    
    logger.info(f"Overlap bounds: left={overlap_left}, right={overlap_right}, bottom={overlap_bottom}, top={overlap_top}")
    
    # 计算目标影像中重叠区域的行列范围
    row_start, col_start = rasterio.transform.rowcol(target_transform, overlap_left, overlap_top)
    row_end, col_end = rasterio.transform.rowcol(target_transform, overlap_right, overlap_bottom)
    
    # 确保索引在有效范围内
    row_start = max(0, row_start)
    col_start = max(0, col_start)
    row_end = min(target_height, row_end + 1)
    col_end = min(target_width, col_end + 1)
    
    output_height = row_end - row_start
    output_width = col_end - col_start
    
    logger.info(f"Output size: {output_height} × {output_width}")
    
    # 计算对齐后的transform（使用目标影像的网格）
    ul_x, ul_y = rasterio.transform.xy(target_transform, row_start, col_start)
    aligned_transform = Affine(
        target_transform[0],
        target_transform[1],
        ul_x,
        target_transform[3],
        target_transform[4],
        ul_y
    )
    
    # 重投影和对齐
    aligned_bands = []
    for band_idx, source_band in enumerate(source_bands, 1):
        aligned_band = np.empty((output_height, output_width), dtype=np.float32)
        aligned_band.fill(np.nan)
        
        warp.reproject(
            source=source_band.astype(np.float32),
            destination=aligned_band,
            src_transform=source_transform,
            src_crs=source_crs,
            dst_transform=aligned_transform,
            dst_crs=target_crs,
            resampling=resampling_method,
            src_nodata=np.nan,
            dst_nodata=np.nan
        )
        
        aligned_bands.append(aligned_band)
        logger.info(f"  Band {band_idx} aligned")
    
    return aligned_bands, aligned_transform, (output_height, output_width)


def extract_dense_pixel_pairs(
    satellite_bands: List[np.ndarray],
    satellite_transform: Affine,
    uav_bands: List[np.ndarray],
    uav_transform: Affine,
    satellite_nodata: Optional[float] = None,
    uav_nodata: Optional[float] = None,
    band_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    从对齐后的影像中提取所有有效像元对，生成密集映射数据。
    
    参数:
        satellite_bands: 卫星影像波段列表（已对齐）
        satellite_transform: 卫星影像的transform
        uav_bands: 无人机影像波段列表（已对齐）
        uav_transform: 无人机影像的transform
        satellite_nodata: 卫星影像的nodata值
        uav_nodata: 无人机影像的nodata值
        band_names: 波段名称列表
    
    返回:
        DataFrame，包含所有有效像元对的特征
    """
    if len(satellite_bands) != len(uav_bands):
        raise ValueError(f"卫星波段数 ({len(satellite_bands)}) 与无人机波段数 ({len(uav_bands)}) 不匹配")
    
    height, width = satellite_bands[0].shape
    
    if band_names is None:
        band_names = [f"band_{i+1}" for i in range(len(satellite_bands))]
    
    # 创建有效像元掩码（排除nodata值）
    valid_mask = np.ones((height, width), dtype=bool)
    
    for sat_band in satellite_bands:
        if satellite_nodata is not None:
            valid_mask = valid_mask & (sat_band != satellite_nodata)
        else:
            valid_mask = valid_mask & ~np.isnan(sat_band)
    
    for uav_band in uav_bands:
        if uav_nodata is not None:
            valid_mask = valid_mask & (uav_band != uav_nodata)
        else:
            valid_mask = valid_mask & ~np.isnan(uav_band)
    
    n_valid = np.sum(valid_mask)
    logger.info(f"Valid pixels: {n_valid} / {height * width} ({100 * n_valid / (height * width):.2f}%)")
    
    # 提取有效像元的坐标和值
    rows, cols = np.where(valid_mask)
    
    # 计算地理坐标
    x_coords, y_coords = rasterio.transform.xy(satellite_transform, rows, cols)
    
    # 提取像元值
    data_dict = {
        'row': rows,
        'col': cols,
        'x': x_coords,
        'y': y_coords
    }
    
    # 添加卫星波段值
    for i, (sat_band, band_name) in enumerate(zip(satellite_bands, band_names)):
        data_dict[f'SAT_{band_name}'] = sat_band[rows, cols]
    
    # 添加无人机波段值
    for i, (uav_band, band_name) in enumerate(zip(uav_bands, band_names)):
        data_dict[f'UAV_{band_name}'] = uav_band[rows, cols]
    
    df = pd.DataFrame(data_dict)
    
    logger.info(f"Extracted {len(df)} pixel pairs")
    
    return df


def prepare_dense_training_data(
    satellite_raster_path: Path,
    uav_raster_path: Path,
    target_resolution: Optional[float] = None,
    satellite_band_indices: Optional[List[int]] = None,
    uav_band_indices: Optional[List[int]] = None,
    satellite_nodata: Optional[float] = None,
    uav_nodata: Optional[float] = None,
    align_to: str = 'satellite'  # 'satellite' or 'uav'
) -> Tuple[pd.DataFrame, Dict]:
    """
    准备密集训练数据：对齐影像并提取所有有效像元对。
    
    参数:
        satellite_raster_path: 卫星影像路径
        uav_raster_path: 无人机影像路径
        target_resolution: 目标分辨率（如果为None，使用卫星影像分辨率）
        satellite_band_indices: 要提取的卫星波段索引（从1开始）
        uav_band_indices: 要提取的无人机波段索引（从1开始）
        satellite_nodata: 卫星影像的nodata值
        uav_nodata: 无人机影像的nodata值
        align_to: 对齐到哪个影像的网格（'satellite' 或 'uav'）
    
    返回:
        DataFrame，包含所有有效像元对
    """
    logger.info("=" * 60)
    logger.info("Preparing Dense Training Data")
    logger.info("=" * 60)
    
    # 读取卫星影像
    with rasterio.open(satellite_raster_path) as sat_src:
        sat_crs = sat_src.crs
        sat_transform = sat_src.transform
        sat_height = sat_src.height
        sat_width = sat_src.width
        
        if satellite_band_indices is None:
            satellite_band_indices = list(range(1, sat_src.count + 1))
        
        satellite_bands = []
        for idx in satellite_band_indices:
            band = sat_src.read(idx)
            satellite_bands.append(band)
        
        if target_resolution is None:
            target_resolution = abs(sat_transform[0])
        
        logger.info(f"Satellite raster: {satellite_raster_path}")
        logger.info(f"  Size: {sat_height} × {sat_width}")
        logger.info(f"  Bands extracted: {satellite_band_indices}")
    
    # 读取无人机影像
    with rasterio.open(uav_raster_path) as uav_src:
        uav_crs = uav_src.crs
        uav_transform = uav_src.transform
        uav_height = uav_src.height
        uav_width = uav_src.width
        
        if uav_band_indices is None:
            uav_band_indices = list(range(1, uav_src.count + 1))
        
        uav_bands_raw = []
        for idx in uav_band_indices:
            band = uav_src.read(idx)
            uav_bands_raw.append(band)
        
        logger.info(f"UAV raster: {uav_raster_path}")
        logger.info(f"  Size: {uav_height} × {uav_width}")
        logger.info(f"  Bands extracted: {uav_band_indices}")
    
    # 如果UAV分辨率高于目标分辨率，先聚合
    uav_resolution = abs(uav_transform[0])
    if uav_resolution < target_resolution:
        logger.info(f"Aggregating UAV from {uav_resolution} to {target_resolution}")
        uav_bands_aggregated = []
        for uav_band in uav_bands_raw:
            aggregated, new_transform = aggregate_to_target_resolution(
                uav_band, uav_transform, target_resolution, uav_nodata
            )
            uav_bands_aggregated.append(aggregated)
        uav_transform = new_transform
        uav_bands = uav_bands_aggregated
    else:
        uav_bands = uav_bands_raw
    
    # 对齐影像到同一网格
    if align_to == 'satellite':
        logger.info("Aligning UAV to satellite grid...")
        # 将UAV对齐到卫星网格
        uav_bands_aligned, aligned_transform, output_shape = align_images_to_grid(
            uav_bands, uav_transform, uav_crs, satellite_raster_path
        )
        
        # 裁剪卫星影像到对齐后的重叠区域
        with rasterio.open(satellite_raster_path) as sat_src:
            # 使用对齐后的transform计算重叠区域
            sat_bounds = sat_src.bounds
            sat_transform = sat_src.transform
            
            # 计算对齐区域在卫星影像中的位置
            row_start, col_start = rasterio.transform.rowcol(sat_transform, aligned_transform[2], aligned_transform[5])
            row_end = row_start + output_shape[0]
            col_end = col_start + output_shape[1]
            
            # 确保在有效范围内
            row_start = max(0, row_start)
            col_start = max(0, col_start)
            row_end = min(sat_src.height, row_end)
            col_end = min(sat_src.width, col_end)
            
            actual_height = row_end - row_start
            actual_width = col_end - col_start
            
            # 裁剪卫星影像
            sat_bands_aligned = []
            for idx in satellite_band_indices:
                band = sat_src.read(idx, window=rasterio.windows.Window(col_start, row_start, actual_width, actual_height))
                # 如果裁剪后的尺寸与对齐后的尺寸不一致，需要调整
                if band.shape != output_shape:
                    # 填充或裁剪到一致尺寸
                    aligned_band = np.full(output_shape, np.nan, dtype=np.float32)
                    h_min = min(band.shape[0], output_shape[0])
                    w_min = min(band.shape[1], output_shape[1])
                    aligned_band[:h_min, :w_min] = band[:h_min, :w_min]
                    sat_bands_aligned.append(aligned_band)
                else:
                    sat_bands_aligned.append(band.astype(np.float32))
    else:
        logger.info("Aligning satellite to UAV grid...")
        # 将卫星对齐到UAV网格
        sat_bands_aligned, aligned_transform, output_shape = align_images_to_grid(
            satellite_bands, sat_transform, sat_crs, uav_raster_path
        )
        
        # UAV影像需要裁剪到对齐区域
        with rasterio.open(uav_raster_path) as uav_src:
            uav_transform_orig = uav_src.transform
            uav_bounds = uav_src.bounds
            
            # 计算对齐区域在UAV影像中的位置
            row_start, col_start = rasterio.transform.rowcol(uav_transform_orig, aligned_transform[2], aligned_transform[5])
            row_end = row_start + output_shape[0]
            col_end = col_start + output_shape[1]
            
            row_start = max(0, row_start)
            col_start = max(0, col_start)
            row_end = min(uav_src.height, row_end)
            col_end = min(uav_src.width, col_end)
            
            actual_height = row_end - row_start
            actual_width = col_end - col_start
            
            # 裁剪UAV影像
            uav_bands_aligned = []
            for idx in uav_band_indices:
                band = uav_src.read(idx, window=rasterio.windows.Window(col_start, row_start, actual_width, actual_height))
                if band.shape != output_shape:
                    aligned_band = np.full(output_shape, np.nan, dtype=np.float32)
                    h_min = min(band.shape[0], output_shape[0])
                    w_min = min(band.shape[1], output_shape[1])
                    aligned_band[:h_min, :w_min] = band[:h_min, :w_min]
                    uav_bands_aligned.append(aligned_band)
                else:
                    uav_bands_aligned.append(band.astype(np.float32))
    
    # 提取密集像元对
    df = extract_dense_pixel_pairs(
        sat_bands_aligned,
        aligned_transform,
        uav_bands_aligned,
        aligned_transform,
        satellite_nodata,
        uav_nodata
    )
    
    # 返回DataFrame和对齐后的影像数据（用于保存）
    aligned_data = {
        'satellite_bands': sat_bands_aligned,
        'uav_bands': uav_bands_aligned,
        'transform': aligned_transform,
        'crs': sat_crs if align_to == 'satellite' else uav_crs,
        'output_shape': output_shape
    }
    
    return df, aligned_data

