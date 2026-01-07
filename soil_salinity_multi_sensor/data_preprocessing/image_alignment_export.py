"""
影像聚合对齐和导出模块

模仿 UAV影像聚合.py 的逻辑，实现：
1. UAV影像空间聚合对齐到Sentinel-2
2. Landsat-8重采样对齐到Sentinel-2
3. 输出对齐后的TIF和CSV配对数据
"""

import numpy as np
import pandas as pd
import rasterio
from rasterio import warp
from rasterio.transform import Affine
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def aggregate_uav_to_s2_resolution(
    uav_band_arrays: List[np.ndarray],
    uav_transform: Affine,
    uav_pixel_size_x: float,
    uav_pixel_size_y: Optional[float] = None,
    target_resolution: float = 8.983152841195699E-05,
    nodata_value: Optional[float] = 65535,
) -> Tuple[List[np.ndarray], Affine]:
    """
    将高分辨率UAV影像聚合成与Sentinel-2分辨率匹配的低分辨率影像。
    
    参数:
        uav_band_arrays: UAV各波段数组列表
        uav_transform: UAV影像的transform
        uav_pixel_size_x: UAV X方向像元分辨率（度）
        uav_pixel_size_y: UAV Y方向像元分辨率（度），如果为None则使用uav_pixel_size_x
        target_resolution: 目标分辨率（度），默认S2分辨率
        nodata_value: nodata值，聚合时会被排除
    
    返回:
        聚合后的波段列表和新的transform
    """
    if uav_pixel_size_y is None:
        uav_pixel_size_y = uav_pixel_size_x
    
    # 计算聚合比例
    factor_x = target_resolution / uav_pixel_size_x
    factor_y = target_resolution / uav_pixel_size_y
    
    if factor_x < 1.0 or factor_y < 1.0:
        raise ValueError(
            f"目标分辨率 ({target_resolution}) 不能小于UAV分辨率 ({uav_pixel_size_x}, {uav_pixel_size_y})"
        )
    
    factor_x_int = int(factor_x)
    factor_y_int = int(factor_y)
    
    aggregated_bands = []
    
    for band_idx, band_array in enumerate(uav_band_arrays, 1):
        h, w = band_array.shape
        
        # 裁剪到能被聚合比例整除的尺寸
        h_new = (h // factor_y_int) * factor_y_int
        w_new = (w // factor_x_int) * factor_x_int
        band_cropped = band_array[:h_new, :w_new]
        
        # 处理nodata值
        if nodata_value is not None:
            band_masked = band_cropped.astype(np.float64)
            band_masked[band_cropped == nodata_value] = np.nan
        else:
            band_masked = band_cropped.astype(np.float64)
        
        # 块聚合：使用reshape + nanmean
        h_out = h_new // factor_y_int
        w_out = w_new // factor_x_int
        
        # 先按行分组
        reshaped_rows = band_masked.reshape(h_out, factor_y_int, w_new)
        row_aggregated = np.nanmean(reshaped_rows, axis=1)
        
        # 再按列reshape
        reshaped_cols = row_aggregated.reshape(h_out, w_out, factor_x_int)
        aggregated = np.nanmean(reshaped_cols, axis=2)
        
        # 将NaN值替换回nodata值
        if nodata_value is not None:
            aggregated = np.where(np.isnan(aggregated), nodata_value, aggregated)
        
        aggregated_bands.append(aggregated.astype(band_array.dtype))
        
        logger.info(
            f"波段 {band_idx}: {band_array.shape} -> {aggregated.shape} "
            f"(factor={factor_x_int:.1f})"
        )
    
    # 更新transform
    new_pixel_size_x = uav_pixel_size_x * factor_x_int
    new_pixel_size_y = uav_pixel_size_y * factor_y_int
    
    new_transform = Affine(
        new_pixel_size_x * (1 if uav_transform[0] >= 0 else -1),
        uav_transform[1],
        uav_transform[2],
        uav_transform[3],
        new_pixel_size_y * (-1 if uav_transform[4] < 0 else 1),
        uav_transform[5],
    )
    
    return aggregated_bands, new_transform


def align_to_s2_grid(
    source_bands: List[np.ndarray],
    source_transform: Affine,
    source_crs: rasterio.crs.CRS,
    s2_raster_path: Path,
    src_nodata: Optional[float] = None,
) -> Tuple[List[np.ndarray], Affine, Tuple[int, int]]:
    """
    将源影像对齐到Sentinel-2的像元网格。
    
    参数:
        source_bands: 源影像波段列表
        source_transform: 源影像的transform
        source_crs: 源影像的CRS
        s2_raster_path: Sentinel-2影像路径
        src_nodata: 源影像的nodata值
    
    返回:
        对齐后的波段列表、新的transform、输出尺寸
    """
    # 读取Sentinel-2影像信息
    with rasterio.open(s2_raster_path) as s2_src:
        s2_transform = s2_src.transform
        s2_crs = s2_src.crs
        s2_height = s2_src.height
        s2_width = s2_src.width
        s2_bounds = s2_src.bounds
    
    logger.info(f"Sentinel-2影像信息:")
    logger.info(f"  尺寸: {s2_height} × {s2_width}")
    logger.info(f"  CRS: {s2_crs}")
    logger.info(f"  Bounds: {s2_bounds}")
    
    # 检查CRS是否一致
    if source_crs != s2_crs:
        logger.warning(f"CRS不一致：源={source_crs}, S2={s2_crs}，将进行重投影")
    
    # 计算源影像bounds
    source_height, source_width = source_bands[0].shape
    source_bounds = rasterio.transform.array_bounds(source_height, source_width, source_transform)
    source_left, source_bottom, source_right, source_top = source_bounds
    
    logger.info(f"源影像bounds: left={source_left:.6f}, bottom={source_bottom:.6f}, "
                f"right={source_right:.6f}, top={source_top:.6f}")
    
    # 计算重叠区域
    overlap_left = max(source_left, s2_bounds.left)
    overlap_right = min(source_right, s2_bounds.right)
    overlap_bottom = max(source_bottom, s2_bounds.bottom)
    overlap_top = min(source_top, s2_bounds.top)
    
    if overlap_left >= overlap_right or overlap_bottom >= overlap_top:
        raise ValueError("源影像与Sentinel-2影像无重叠区域")
    
    logger.info(f"重叠区域: left={overlap_left:.6f}, right={overlap_right:.6f}, "
                f"bottom={overlap_bottom:.6f}, top={overlap_top:.6f}")
    
    # 计算在Sentinel-2网格中的输出窗口
    row_start, col_start = rasterio.transform.rowcol(
        s2_transform, overlap_left, overlap_top
    )
    row_end, col_end = rasterio.transform.rowcol(
        s2_transform, overlap_right, overlap_bottom
    )
    
    # 确保索引在有效范围内
    row_start = max(0, row_start)
    col_start = max(0, col_start)
    row_end = min(s2_height, row_end + 1)
    col_end = min(s2_width, col_end + 1)
    
    output_height = row_end - row_start
    output_width = col_end - col_start
    
    logger.info(f"输出尺寸: {output_height} × {output_width}")
    logger.info(f"在Sentinel-2中的位置: 行 [{row_start}, {row_end}), 列 [{col_start}, {col_end})")
    
    # 对齐到Sentinel-2的像元网格
    s2_ul_x = s2_transform[2] + col_start * s2_transform[0]
    s2_ul_y = s2_transform[5] + row_start * s2_transform[4]
    
    aligned_transform = Affine(
        s2_transform[0],
        s2_transform[1],
        s2_ul_x,
        s2_transform[3],
        s2_transform[4],
        s2_ul_y,
    )
    
    logger.info(f"对齐后的transform: {aligned_transform}")
    
    # 使用rasterio的重投影功能将源数据重采样到Sentinel-2网格
    aligned_bands = []
    
    for band_idx, source_band in enumerate(source_bands, 1):
        # 创建目标数组（使用float32以支持NaN）
        aligned_band = np.empty((output_height, output_width), dtype=np.float32)
        aligned_band.fill(np.nan)
        
        # 重投影
        warp.reproject(
            source=source_band.astype(np.float32),
            destination=aligned_band,
            src_transform=source_transform,
            src_crs=source_crs,
            dst_transform=aligned_transform,
            dst_crs=s2_crs,
            resampling=warp.Resampling.nearest,
            src_nodata=src_nodata if src_nodata is not None else np.nan,
            dst_nodata=np.nan,
        )
        
        aligned_bands.append(aligned_band)
        logger.info(f"波段 {band_idx} 已对齐到Sentinel-2网格")
    
    return aligned_bands, aligned_transform, (output_height, output_width)


def extract_pixel_pairs(
    aligned_bands: List[np.ndarray],
    aligned_transform: Affine,
    s2_raster_path: Path,
    band_prefix: str = "UAV",
    aligned_nodata: Optional[float] = None,
    s2_nodata: Optional[float] = None,
) -> pd.DataFrame:
    """
    构建对齐后的像元值与Sentinel-2像元值的配对数据。
    
    参数:
        aligned_bands: 对齐后的波段列表
        aligned_transform: 对齐后的transform
        s2_raster_path: Sentinel-2影像路径
        band_prefix: 波段列名前缀（如"UAV"或"L8"）
        aligned_nodata: 对齐影像的nodata值
        s2_nodata: Sentinel-2的nodata值
    
    返回:
        DataFrame，包含配对数据（已排除nodata值）
    """
    if not aligned_bands:
        raise ValueError("对齐波段数组不能为空")
    
    # 获取对齐影像尺寸
    aligned_height, aligned_width = aligned_bands[0].shape
    
    logger.info(f"对齐影像尺寸: {aligned_height} × {aligned_width}")
    
    # 读取Sentinel-2影像
    with rasterio.open(s2_raster_path) as s2_src:
        s2_transform = s2_src.transform
        s2_height = s2_src.height
        s2_width = s2_src.width
        s2_num_bands = s2_src.count
        
        logger.info(f"Sentinel-2影像尺寸: {s2_height} × {s2_width}, 波段数: {s2_num_bands}")
        
        # 计算对齐影像左上角在Sentinel-2中的行列位置
        aligned_ul_x = aligned_transform[2]
        aligned_ul_y = aligned_transform[5]
        
        row_start, col_start = rasterio.transform.rowcol(s2_transform, aligned_ul_x, aligned_ul_y)
        
        # 确保索引在有效范围内
        row_start = max(0, row_start)
        col_start = max(0, col_start)
        row_end = min(s2_height, row_start + aligned_height)
        col_end = min(s2_width, col_start + aligned_width)
        
        # 如果超出范围，调整对齐数据尺寸
        actual_height = row_end - row_start
        actual_width = col_end - col_start
        
        if actual_height != aligned_height or actual_width != aligned_width:
            logger.warning(f"对齐影像超出Sentinel-2范围，裁剪到: {actual_height} × {actual_width}")
            aligned_bands = [band[:actual_height, :actual_width] for band in aligned_bands]
            aligned_height, aligned_width = actual_height, actual_width
        
        logger.info(f"在Sentinel-2中的位置: 行 [{row_start}, {row_end}), 列 [{col_start}, {col_end})")
        
        # 读取Sentinel-2对应区域的各波段数据
        s2_window = rasterio.windows.Window.from_slices(
            (row_start, row_end),
            (col_start, col_end)
        )
        
        s2_bands = []
        for band_idx in range(1, s2_num_bands + 1):
            s2_band = s2_src.read(band_idx, window=s2_window)
            s2_bands.append(s2_band)
            logger.info(f"读取Sentinel-2波段 {band_idx}: shape = {s2_band.shape}")
    
    # 构建配对数据
    data_dict = {}
    
    # 添加对齐后的波段数据
    for band_idx, aligned_band in enumerate(aligned_bands, 1):
        data_dict[f"{band_prefix}_band_{band_idx}"] = aligned_band.flatten()
    
    # 添加Sentinel-2波段数据
    for band_idx, s2_band in enumerate(s2_bands, 1):
        # 确保尺寸匹配
        if s2_band.shape != (aligned_height, aligned_width):
            s2_band = s2_band[:aligned_height, :aligned_width]
        data_dict[f"S2_band_{band_idx}"] = s2_band.flatten()
    
    # 构建DataFrame
    df = pd.DataFrame(data_dict)
    
    # 过滤掉包含nodata值的行（同时检查对齐影像和S2）
    initial_count = len(df)
    
    # 获取列名
    aligned_cols = [col for col in df.columns if col.startswith(f"{band_prefix}_band_")]
    s2_cols = [col for col in df.columns if col.startswith("S2_band_")]
    
    # 创建过滤掩码
    mask = pd.Series(True, index=df.index)
    
    # 过滤对齐影像的nodata值
    if aligned_nodata is not None:
        logger.info(f"过滤 {band_prefix} nodata值: {aligned_nodata}")
        for col in aligned_cols:
            mask = mask & (df[col] != aligned_nodata) & (~pd.isna(df[col]))
    else:
        # 如果没有指定nodata，过滤NaN值
        for col in aligned_cols:
            mask = mask & (~pd.isna(df[col]))
    
    # 过滤Sentinel-2的nodata值
    if s2_nodata is not None:
        logger.info(f"过滤Sentinel-2 nodata值: {s2_nodata}")
        for col in s2_cols:
            mask = mask & (df[col] != s2_nodata) & (~pd.isna(df[col]))
    else:
        # 如果没有指定nodata，过滤NaN值
        for col in s2_cols:
            mask = mask & (~pd.isna(df[col]))
    
    # 应用过滤
    df = df[mask].copy()
    filtered_count = initial_count - len(df)
    
    if filtered_count > 0:
        logger.info(f"已过滤掉 {filtered_count} 行包含nodata值的数据")
        logger.info(f"  过滤前: {initial_count} 行")
        logger.info(f"  过滤后: {len(df)} 行")
        logger.info(f"  过滤率: {filtered_count / initial_count * 100:.2f}%")
    
    logger.info(f"配对数据构建完成:")
    logger.info(f"  总像元数（过滤后）: {len(df)}")
    logger.info(f"  列数: {len(df.columns)}")
    logger.info(f"  {band_prefix}波段数: {len(aligned_bands)}")
    logger.info(f"  Sentinel-2波段数: {len(s2_bands)}")
    
    return df


def save_aligned_raster(
    aligned_bands: List[np.ndarray],
    output_path: Path,
    transform: Affine,
    crs: rasterio.crs.CRS,
    dtype: str = "float32",
    nodata: Optional[float] = None,
) -> None:
    """
    保存对齐后的影像为多波段GeoTIFF。
    
    参数:
        aligned_bands: 对齐后的波段列表
        output_path: 输出GeoTIFF文件路径
        transform: 对齐后的transform
        crs: 坐标参考系统
        dtype: 输出数据类型
        nodata: nodata值
    """
    if not aligned_bands:
        raise ValueError("对齐波段数组不能为空")
    
    # 获取输出尺寸
    height, width = aligned_bands[0].shape
    num_bands = len(aligned_bands)
    
    # 检查所有波段尺寸是否一致
    for i, band in enumerate(aligned_bands):
        if band.shape != (height, width):
            raise ValueError(
                f"波段 {i + 1} 的尺寸 {band.shape} 与其他波段不一致 ({height}, {width})"
            )
    
    # 自动设置nodata值
    if nodata is None:
        if dtype.startswith("float"):
            nodata = -9999.0
        elif dtype.startswith("uint"):
            nodata = 0
        elif dtype.startswith("int"):
            nodata = -9999
        else:
            nodata = -9999.0
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 写入GeoTIFF
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=num_bands,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="lzw",
    ) as dst:
        for band_idx, band_array in enumerate(aligned_bands, 1):
            # 处理NaN值
            if np.issubdtype(band_array.dtype, np.integer):
                band_array = band_array.astype(np.float32)
            
            # 将NaN值转换为nodata值
            if np.isnan(band_array).any():
                band_array = np.where(np.isnan(band_array), nodata, band_array)
                logger.info(f"波段 {band_idx}: 已将NaN值转换为nodata ({nodata})")
            
            # 转换到目标数据类型
            if dtype != str(band_array.dtype):
                band_array = band_array.astype(dtype)
            
            dst.write(band_array, band_idx)
            logger.info(f"波段 {band_idx}/{num_bands} 已写入")
    
    logger.info(f"对齐影像GeoTIFF已保存：{output_path}")
    logger.info(f"  波段数: {num_bands}")
    logger.info(f"  尺寸: {height} × {width}")
    logger.info(f"  数据类型: {dtype}")
    logger.info(f"  Nodata: {nodata}")


def process_uav_alignment(
    uav_raster_path: Path,
    s2_raster_path: Path,
    output_tif_path: Path,
    output_csv_path: Path,
    uav_nodata: Optional[float] = 65535,
    s2_nodata: Optional[float] = 0,
    target_resolution: float = 8.983152841195699E-05,
) -> None:
    """
    处理UAV影像聚合对齐到S2的完整流程。
    
    参数:
        uav_raster_path: UAV影像路径
        s2_raster_path: Sentinel-2影像路径
        output_tif_path: 输出TIF路径
        output_csv_path: 输出CSV路径
        uav_nodata: UAV的nodata值
        s2_nodata: Sentinel-2的nodata值
        target_resolution: 目标分辨率（度）
    """
    logger.info("=" * 80)
    logger.info("UAV影像聚合对齐到Sentinel-2处理流程")
    logger.info("=" * 80)
    
    # 步骤1: 读取UAV影像
    logger.info("\n步骤 1/5: 读取UAV影像")
    with rasterio.open(uav_raster_path) as src:
        height = src.height
        width = src.width
        num_bands = src.count
        transform = src.transform
        crs = src.crs
        
        pixel_size_x = abs(transform[0])
        pixel_size_y = abs(transform[4])
        
        bands = []
        for band_idx in range(1, num_bands + 1):
            band_data = src.read(band_idx)
            bands.append(band_data)
            logger.info(f"波段 {band_idx}: shape = {band_data.shape}, dtype = {band_data.dtype}")
    
    logger.info(f"UAV影像信息:")
    logger.info(f"  波段数: {num_bands}")
    logger.info(f"  尺寸: {height} × {width}")
    logger.info(f"  像元分辨率 X: {pixel_size_x}")
    logger.info(f"  像元分辨率 Y: {pixel_size_y}")
    logger.info(f"  CRS: {crs}")
    
    # 步骤2: 空间聚合
    logger.info("\n步骤 2/5: 空间聚合到Sentinel-2分辨率")
    logger.info(f"目标分辨率: {target_resolution}")
    
    aggregated_bands, new_transform = aggregate_uav_to_s2_resolution(
        uav_band_arrays=bands,
        uav_transform=transform,
        uav_pixel_size_x=pixel_size_x,
        uav_pixel_size_y=pixel_size_y,
        target_resolution=target_resolution,
        nodata_value=uav_nodata,
    )
    
    # 步骤3: 对齐到Sentinel-2网格
    logger.info("\n步骤 3/5: 对齐到Sentinel-2像元网格")
    
    aligned_bands, s2_aligned_transform, output_shape = align_to_s2_grid(
        source_bands=aggregated_bands,
        source_transform=new_transform,
        source_crs=crs,
        s2_raster_path=s2_raster_path,
        src_nodata=uav_nodata,
    )
    
    # 步骤4: 保存对齐后的GeoTIFF
    logger.info("\n步骤 4/5: 保存对齐后的GeoTIFF")
    
    with rasterio.open(s2_raster_path) as s2_src:
        s2_crs = s2_src.crs
    
    save_aligned_raster(
        aligned_bands=aligned_bands,
        output_path=output_tif_path,
        transform=s2_aligned_transform,
        crs=s2_crs,
        dtype="float32",
        nodata=None,
    )
    
    # 步骤5: 提取像元配对数据
    logger.info("\n步骤 5/5: 提取像元配对数据")
    
    pixel_pairs_df = extract_pixel_pairs(
        aligned_bands=aligned_bands,
        aligned_transform=s2_aligned_transform,
        s2_raster_path=s2_raster_path,
        band_prefix="UAV",
        aligned_nodata=uav_nodata,
        s2_nodata=s2_nodata,
    )
    
    # 保存配对数据为CSV
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    pixel_pairs_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    
    logger.info(f"配对数据已保存：{output_csv_path}")
    logger.info(f"  数据行数: {len(pixel_pairs_df)}")
    logger.info(f"  数据列数: {len(pixel_pairs_df.columns)}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ UAV影像聚合对齐处理完成！")
    logger.info("=" * 80)


def process_l8_alignment(
    l8_raster_path: Path,
    s2_raster_path: Path,
    output_tif_path: Path,
    output_csv_path: Path,
    l8_nodata: Optional[float] = None,
    s2_nodata: Optional[float] = 0,
) -> None:
    """
    处理Landsat-8影像重采样对齐到S2的完整流程。
    
    参数:
        l8_raster_path: Landsat-8影像路径
        s2_raster_path: Sentinel-2影像路径
        output_tif_path: 输出TIF路径
        output_csv_path: 输出CSV路径
        l8_nodata: Landsat-8的nodata值
        s2_nodata: Sentinel-2的nodata值
    """
    logger.info("=" * 80)
    logger.info("Landsat-8影像重采样对齐到Sentinel-2处理流程")
    logger.info("=" * 80)
    
    # 步骤1: 读取Landsat-8影像
    logger.info("\n步骤 1/4: 读取Landsat-8影像")
    with rasterio.open(l8_raster_path) as src:
        height = src.height
        width = src.width
        num_bands = src.count
        transform = src.transform
        crs = src.crs
        
        bands = []
        for band_idx in range(1, num_bands + 1):
            band_data = src.read(band_idx)
            bands.append(band_data)
            logger.info(f"波段 {band_idx}: shape = {band_data.shape}, dtype = {band_data.dtype}")
    
    logger.info(f"Landsat-8影像信息:")
    logger.info(f"  波段数: {num_bands}")
    logger.info(f"  尺寸: {height} × {width}")
    logger.info(f"  CRS: {crs}")
    
    # 步骤2: 对齐到Sentinel-2网格（直接重采样，不需要聚合）
    logger.info("\n步骤 2/4: 对齐到Sentinel-2像元网格")
    
    aligned_bands, s2_aligned_transform, output_shape = align_to_s2_grid(
        source_bands=bands,
        source_transform=transform,
        source_crs=crs,
        s2_raster_path=s2_raster_path,
        src_nodata=l8_nodata,
    )
    
    # 步骤3: 保存对齐后的GeoTIFF
    logger.info("\n步骤 3/4: 保存对齐后的GeoTIFF")
    
    with rasterio.open(s2_raster_path) as s2_src:
        s2_crs = s2_src.crs
    
    save_aligned_raster(
        aligned_bands=aligned_bands,
        output_path=output_tif_path,
        transform=s2_aligned_transform,
        crs=s2_crs,
        dtype="float32",
        nodata=None,
    )
    
    # 步骤4: 提取像元配对数据
    logger.info("\n步骤 4/4: 提取像元配对数据")
    
    pixel_pairs_df = extract_pixel_pairs(
        aligned_bands=aligned_bands,
        aligned_transform=s2_aligned_transform,
        s2_raster_path=s2_raster_path,
        band_prefix="L8",
        aligned_nodata=l8_nodata,
        s2_nodata=s2_nodata,
    )
    
    # 保存配对数据为CSV
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    pixel_pairs_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    
    logger.info(f"配对数据已保存：{output_csv_path}")
    logger.info(f"  数据行数: {len(pixel_pairs_df)}")
    logger.info(f"  数据列数: {len(pixel_pairs_df.columns)}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ Landsat-8影像重采样对齐处理完成！")
    logger.info("=" * 80)

