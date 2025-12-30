"""
================================================================================
代码功能：UAV影像聚合生成仿卫星像元（Satellite Simulation）
================================================================================

【核心目的】
将高分辨率无人机（UAV）多光谱影像聚合成与卫星（Sentinel-2）分辨率匹配的
"仿卫星像元"数据，用于星机光谱融合反演实验。通过空间聚合和网格对齐，
实现UAV与Sentinel-2数据的像元级配对，为后续的光谱融合建模提供训练数据。

【主要功能模块】

1. 影像读取与信息提取
   - 读取UAV多光谱影像（G、R、REG、NIR波段）
   - 读取Sentinel-2多光谱影像
   - 提取影像基本信息：尺寸、分辨率、坐标系统、变换矩阵等

2. 空间聚合（aggregate_to_satellite_resolution）
   - 将高分辨率UAV影像按目标分辨率进行块均值聚合
   - 使用nanmean方法排除nodata值（65535）的影响
   - 保持地理坐标系统不变，仅改变像元大小
   - 支持多波段独立处理，可设置不同波段的目标分辨率

3. 网格对齐（align_to_sentinel2_grid）
   - 将聚合后的UAV数据对齐到Sentinel-2的像元网格
   - 使用rasterio的warp.reproject进行重投影和对齐
   - 处理CRS不一致的情况（自动重投影）
   - 计算重叠区域，确保数据空间一致性

4. 像元配对数据提取（extract_pixel_pairs）
   - 构建仿卫星像元值与Sentinel-2像元值的配对数据
   - 逐像元提取对应位置的波段值
   - 自动过滤包含nodata值的像元（✨已修复：同时过滤UAV和S2的nodata）
   - 生成CSV格式的配对数据集，用于后续建模

5. 结果保存
   - 保存聚合对齐后的仿卫星像元为多波段GeoTIFF
   - 保存像元配对数据为CSV文件

【技术实现】

使用的核心库：
- rasterio: 栅格数据读写、坐标变换、重投影
- numpy: 数组操作、块聚合计算（reshape + nanmean）
- pandas: 配对数据管理和CSV输出

关键技术方法：
1. 块均值聚合算法：
   - 使用reshape将高分辨率像元重组为块结构
   - 使用nanmean对每个块求均值，自动排除nodata值
   - 保持聚合后的像元数量与目标分辨率匹配

2. 网格对齐算法：
   - 计算UAV聚合数据与Sentinel-2的重叠区域
   - 使用rasterio.transform.rowcol计算行列索引
   - 使用warp.reproject进行最近邻重采样，避免插值nodata值

3. 坐标系统处理：
   - 自动检测CRS一致性
   - 支持不同CRS之间的自动重投影
   - 保持地理坐标精度

【处理流程】

输入：
├── UAV多光谱影像（高分辨率，如0.05m）
└── Sentinel-2多光谱影像（低分辨率，如10m）

处理步骤：
1. 读取UAV影像 → 提取波段数据和元信息
2. 空间聚合 → 按目标分辨率（如10m）进行块均值聚合
3. 网格对齐 → 对齐到Sentinel-2的像元网格
4. 像元配对 → 提取对应位置的像元值
5. 数据过滤 → 排除nodata值（✨已修复：同时检查UAV和S2）
6. 结果保存 → 输出GeoTIFF和CSV

输出：
├── 仿卫星像元GeoTIFF（与Sentinel-2分辨率匹配）
└── 像元配对CSV（SAT_sim_band* 和 S2_band* 列）

【应用场景】

本代码主要用于"星机光谱融合反演"实验：
- 解决UAV与卫星数据分辨率不匹配的问题
- 生成用于训练光谱映射模型的配对数据
- 实现高分辨率UAV数据向卫星分辨率的尺度转换
- 为后续的盐分反演融合建模提供数据基础

【注意事项】

1. nodata值处理：默认使用65535作为nodata值，聚合时会自动排除
2. 分辨率匹配：确保目标分辨率大于等于UAV原始分辨率
3. 空间范围：UAV数据需要与Sentinel-2有重叠区域
4. 数据类型：输出使用float32以保留聚合后的浮点精度
5. 内存占用：大影像处理时注意内存使用情况

【版本更新】
v1.1 - 修复nodata过滤逻辑，同时检查UAV和Sentinel-2的nodata值

================================================================================
作者：根据富平盐分反演项目需求开发
日期：2024-2025
================================================================================
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio import warp
from rasterio.transform import Affine


# ===============================================================================
#                          用户配置区域
# ===============================================================================

UAV_RASTER_PATH = Path(
    r"D:\富平星机光谱融合反演\数据\multi_G_R_REG_NIR.tif"
)

SENTINEL2_RASTER_PATH = Path(
    r"D:\富平星机光谱融合反演\数据\Fuping_Sentinel2_20240830.tif"
)

OUTPUT_ALIGNED_TIFF = Path(
    r"D:\富平星机光谱融合反演\数据\satellite_simulation_aligned.tif"
)

OUTPUT_PIXEL_PAIRS_CSV = Path(
    r"D:\富平星机光谱融合反演\数据\pixel_pairs.csv"
)

# 目标分辨率（单位：度）
TARGET_RESOLUTION = 8.983152841195699E-05

# Nodata值设置
UAV_NODATA = 65535
S2_NODATA = 0  # Sentinel-2的nodata值，如果没有则设为None


# ===============================================================================
#                          核心功能函数
# ===============================================================================

def aggregate_to_satellite_resolution(
    uav_band_arrays: list,
    uav_transform: Affine,
    uav_pixel_size_x: float,
    uav_pixel_size_y: float | None = None,
    target_resolution: float | list[float] | None = None,
    nodata_value: float | int | None = 65535,
) -> tuple[list[np.ndarray], Affine]:
    """
    将高分辨率 UAV 影像聚合成与卫星分辨率匹配的低分辨率"仿卫星像元"
    
    参数:
        uav_band_arrays: list，每个元素是一个 numpy 2D array（UAV 各波段）
        uav_transform: rasterio.transform，UAV 影像的 transform
        uav_pixel_size_x: float，UAV X 方向像元分辨率（单位：度，如果是地理坐标系）
        uav_pixel_size_y: float | None，UAV Y 方向像元分辨率，如果为 None 则使用 uav_pixel_size_x
        target_resolution: float 或 list[float]，目标分辨率（单位：度）
                          如果是 list，则每个波段使用对应的分辨率
        nodata_value: float | int | None，nodata 值，在聚合时会被排除，默认为 65535
    
    返回:
        sat_sim_bands: list，包含各波段聚合后的 numpy 数组
        new_transform: Affine，根据聚合比例更新后的 transform
    """
    # 如果 Y 方向分辨率未提供，使用 X 方向分辨率
    if uav_pixel_size_y is None:
        uav_pixel_size_y = uav_pixel_size_x
    
    # 如果 target_resolution 是单个值，转换为列表
    if isinstance(target_resolution, (int, float)):
        target_resolutions = [target_resolution] * len(uav_band_arrays)
    else:
        target_resolutions = target_resolution
    
    if len(target_resolutions) != len(uav_band_arrays):
        raise ValueError(
            f"目标分辨率数量 ({len(target_resolutions)}) 与波段数量 ({len(uav_band_arrays)}) 不匹配"
        )
    
    # 使用 X 方向分辨率计算聚合比例（假设使用相同的聚合比例）
    first_factor = target_resolutions[0] / uav_pixel_size_x
    factor_int = int(first_factor)
    
    sat_sim_bands = []
    
    for band_idx, (band_array, target_res) in enumerate(
        zip(uav_band_arrays, target_resolutions)
    ):
        # 计算聚合比例（使用 X 方向分辨率）
        factor = target_res / uav_pixel_size_x
        
        if factor < 1.0:
            raise ValueError(
                f"波段 {band_idx + 1}: 目标分辨率 ({target_res}) 不能小于 UAV 分辨率 ({uav_pixel_size_x})"
            )
        
        # 将 factor 转换为整数（向下取整）
        band_factor_int = int(factor)
        
        # 获取原始尺寸
        h, w = band_array.shape
        
        # 裁剪到能被 band_factor_int 整除的尺寸
        h_new = (h // band_factor_int) * band_factor_int
        w_new = (w // band_factor_int) * band_factor_int
        
        # 裁剪数组
        band_cropped = band_array[:h_new, :w_new]
        
        # 创建掩码，排除 nodata 值
        if nodata_value is not None:
            # 将 nodata 值替换为 NaN，然后使用 nanmean
            band_masked = band_cropped.astype(np.float64)
            band_masked[band_cropped == nodata_value] = np.nan
        else:
            band_masked = band_cropped.astype(np.float64)
        
        # 使用 reshape + nanmean 进行 block mean 聚合（排除 nodata 值）
        # 将数组重塑为 (h_out, band_factor_int, w_out, band_factor_int)，然后对块求均值
        h_out = h_new // band_factor_int
        w_out = w_new // band_factor_int
        
        # 方法：使用 nanmean 排除 nodata 值
        # 1. 先按行分组: (h_out, band_factor_int, w_new)
        reshaped_rows = band_masked.reshape(h_out, band_factor_int, w_new)
        # 2. 对行方向求均值（排除 NaN/nodata）: (h_out, w_new)
        row_aggregated = np.nanmean(reshaped_rows, axis=1)
        # 3. 再按列 reshape: (h_out, w_out, band_factor_int)
        reshaped_cols = row_aggregated.reshape(h_out, w_out, band_factor_int)
        # 4. 对列方向求均值（排除 NaN/nodata）: (h_out, w_out)
        aggregated = np.nanmean(reshaped_cols, axis=2)
        
        # 将 NaN 值替换回 nodata 值
        if nodata_value is not None:
            aggregated = np.where(np.isnan(aggregated), nodata_value, aggregated)
        
        # 转换回原始数据类型
        aggregated = aggregated.astype(band_array.dtype)
        
        sat_sim_bands.append(aggregated)
        
        print(
            f"波段 {band_idx + 1}: {band_array.shape} -> {aggregated.shape} "
            f"(factor={band_factor_int:.1f})"
        )
    
    # 更新 transform
    # 使用统一的聚合比例（使用第一个波段的 factor）
    # 新的像元大小 = 原始像元大小 × factor
    new_pixel_size_x = uav_pixel_size_x * factor_int
    new_pixel_size_y = uav_pixel_size_y * factor_int
    
    # 更新 transform（保持左上角坐标不变，更新像元大小）
    new_transform = Affine(
        new_pixel_size_x * (1 if uav_transform[0] >= 0 else -1),
        uav_transform[1],
        uav_transform[2],
        uav_transform[3],
        new_pixel_size_y * (-1 if uav_transform[4] < 0 else 1),
        uav_transform[5],
    )
    
    return sat_sim_bands, new_transform


def align_to_sentinel2_grid(
    uav_aggregated_bands: list[np.ndarray],
    uav_transform: Affine,
    uav_crs: rasterio.crs.CRS,
    sentinel2_path: Path,
) -> tuple[list[np.ndarray], Affine, tuple[int, int]]:
    """
    将 UAV 聚合后的数据对齐到 Sentinel-2 影像的像元网格
    
    参数:
        uav_aggregated_bands: list，UAV 聚合后的各波段 numpy 数组
        uav_transform: Affine，UAV 聚合后的 transform
        uav_crs: rasterio.crs.CRS，UAV 影像的 CRS
        sentinel2_path: Path，Sentinel-2 影像路径
    
    返回:
        aligned_bands: list，对齐后的各波段 numpy 数组
        s2_transform: Affine，Sentinel-2 的 transform
        output_shape: tuple[int, int]，输出尺寸 (height, width)
    """
    # 读取 Sentinel-2 影像信息
    with rasterio.open(sentinel2_path) as s2_src:
        s2_transform = s2_src.transform
        s2_crs = s2_src.crs
        s2_height = s2_src.height
        s2_width = s2_src.width
        s2_bounds = s2_src.bounds
        
        # 计算 Sentinel-2 的像元大小
        s2_pixel_size_x = abs(s2_transform[0])
        s2_pixel_size_y = abs(s2_transform[4])
        
        print("\n" + "=" * 60)
        print("Sentinel-2 影像信息")
        print("=" * 60)
        print(f"影像路径: {sentinel2_path}")
        print(f"尺寸: {s2_height} × {s2_width}")
        print(f"像元大小 X: {s2_pixel_size_x:.10f}")
        print(f"像元大小 Y: {s2_pixel_size_y:.10f}")
        print(f"Transform: {s2_transform}")
        print(f"CRS: {s2_crs}")
        print(f"Bounds: {s2_bounds}")
        print("=" * 60)
    
    # 检查 CRS 是否一致
    if uav_crs != s2_crs:
        print(f"⚠️ CRS 不一致：UAV={uav_crs}, S2={s2_crs}")
        print("   将对 UAV 数据进行重投影...")
    
    # 计算 UAV 聚合数据的 bounds
    uav_height, uav_width = uav_aggregated_bands[0].shape
    uav_bounds = rasterio.transform.array_bounds(uav_height, uav_width, uav_transform)
    uav_left, uav_bottom, uav_right, uav_top = uav_bounds
    
    print(f"\n📊 UAV 聚合数据 bounds: left={uav_left:.6f}, bottom={uav_bottom:.6f}, "
          f"right={uav_right:.6f}, top={uav_top:.6f}")
    print(f"📊 Sentinel-2 bounds: left={s2_bounds.left:.6f}, bottom={s2_bounds.bottom:.6f}, "
          f"right={s2_bounds.right:.6f}, top={s2_bounds.top:.6f}")
    
    # 计算重叠区域
    overlap_left = max(uav_left, s2_bounds.left)
    overlap_right = min(uav_right, s2_bounds.right)
    overlap_bottom = max(uav_bottom, s2_bounds.bottom)
    overlap_top = min(uav_top, s2_bounds.top)
    
    if overlap_left >= overlap_right or overlap_bottom >= overlap_top:
        raise ValueError("UAV 聚合数据与 Sentinel-2 影像无重叠区域")
    
    print(f"📊 重叠区域: left={overlap_left:.6f}, right={overlap_right:.6f}, "
          f"bottom={overlap_bottom:.6f}, top={overlap_top:.6f}")
    
    # 计算在 Sentinel-2 网格中的输出窗口
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
    
    print(f"📊 输出尺寸: {output_height} × {output_width}")
    print(f"📊 在 Sentinel-2 中的位置: 行 [{row_start}, {row_end}), 列 [{col_start}, {col_end})")
    
    # 对齐到 Sentinel-2 的像元网格
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
    
    print(f"📊 对齐后的 transform: {aligned_transform}")
    
    # 使用 rasterio 的重投影功能将 UAV 数据重采样到 Sentinel-2 网格
    aligned_bands = []
    
    for band_idx, uav_band in enumerate(uav_aggregated_bands, start=1):
        # 创建目标数组（使用 float32 以支持 NaN）
        aligned_band = np.empty((output_height, output_width), dtype=np.float32)
        aligned_band.fill(np.nan)
        
        # 重投影，指定 nodata 值以避免插值 65535
        warp.reproject(
            source=uav_band.astype(np.float32),
            destination=aligned_band,
            src_transform=uav_transform,
            src_crs=uav_crs,
            dst_transform=aligned_transform,
            dst_crs=s2_crs,
            resampling=warp.Resampling.nearest,
            src_nodata=65535,
            dst_nodata=np.nan,
        )
        
        aligned_bands.append(aligned_band)
        print(f"✅ 波段 {band_idx} 已对齐到 Sentinel-2 网格")
    
    return aligned_bands, aligned_transform, (output_height, output_width)


def extract_pixel_pairs(
    sat_sim_bands: list[np.ndarray],
    sat_sim_transform: Affine,
    sentinel2_path: Path,
    sat_nodata: float | int | None = 65535,
    s2_nodata: float | int | None = None,
) -> pd.DataFrame:
    """
    构建仿卫星像元值与 Sentinel-2 像元值的配对数据
    
    ✨ 修复说明：现在会同时检查 UAV 和 Sentinel-2 的 nodata 值
    
    参数:
        sat_sim_bands: list，对齐到 Sentinel-2 网格的仿卫星各波段 numpy 数组
        sat_sim_transform: Affine，仿卫星像元的 transform（应该与 Sentinel-2 网格对齐）
        sentinel2_path: Path，Sentinel-2 影像路径
        sat_nodata: float | int | None，UAV仿卫星的 nodata 值，默认为 65535
        s2_nodata: float | int | None，Sentinel-2 的 nodata 值，如果为 None 则不过滤
    
    返回:
        pd.DataFrame，包含 SAT_sim_band1, SAT_sim_band2, ... 
        和 S2_band1, S2_band2, ... 列（已排除包含 nodata 的行）
    """
    if not sat_sim_bands:
        raise ValueError("仿卫星波段数组不能为空")
    
    # 获取仿卫星影像尺寸
    sat_height, sat_width = sat_sim_bands[0].shape
    
    print(f"\n📊 仿卫星影像尺寸: {sat_height} × {sat_width}")
    
    # 读取 Sentinel-2 影像
    with rasterio.open(sentinel2_path) as s2_src:
        s2_transform = s2_src.transform
        s2_height = s2_src.height
        s2_width = s2_src.width
        s2_num_bands = s2_src.count
        
        print(f"📊 Sentinel-2 影像尺寸: {s2_height} × {s2_width}, 波段数: {s2_num_bands}")
        print(f"📊 Sentinel-2 Transform: {s2_transform}")
        print(f"📊 仿卫星 Transform: {sat_sim_transform}")
        
        # 检查 transform 是否匹配
        transform_diff = abs(sat_sim_transform[0] - s2_transform[0]) + abs(sat_sim_transform[4] - s2_transform[4])
        if transform_diff > 1e-10:
            print(f"⚠️ Transform 不完全匹配，差异: {transform_diff:.2e}")
        
        # 计算仿卫星影像左上角在 Sentinel-2 中的行列位置
        sat_ul_x = sat_sim_transform[2]
        sat_ul_y = sat_sim_transform[5]
        
        row_start, col_start = rasterio.transform.rowcol(s2_transform, sat_ul_x, sat_ul_y)
        
        # 确保索引在有效范围内
        row_start = max(0, row_start)
        col_start = max(0, col_start)
        row_end = min(s2_height, row_start + sat_height)
        col_end = min(s2_width, col_start + sat_width)
        
        # 如果超出范围，调整仿卫星数据尺寸
        actual_height = row_end - row_start
        actual_width = col_end - col_start
        
        if actual_height != sat_height or actual_width != sat_width:
            print(f"⚠️ 仿卫星影像超出 Sentinel-2 范围，裁剪到: {actual_height} × {actual_width}")
            sat_sim_bands = [band[:actual_height, :actual_width] for band in sat_sim_bands]
            sat_height, sat_width = actual_height, actual_width
        
        print(f"📊 在 Sentinel-2 中的位置: 行 [{row_start}, {row_end}), 列 [{col_start}, {col_end})")
        
        # 读取 Sentinel-2 对应区域的各波段数据
        s2_window = rasterio.windows.Window.from_slices(
            (row_start, row_end),
            (col_start, col_end)
        )
        
        s2_bands = []
        for band_idx in range(1, s2_num_bands + 1):
            s2_band = s2_src.read(band_idx, window=s2_window)
            s2_bands.append(s2_band)
            print(f"✅ 读取 Sentinel-2 波段 {band_idx}: shape = {s2_band.shape}")
    
    # 构建配对数据
    data_dict = {}
    
    # 添加仿卫星波段数据
    for band_idx, sat_band in enumerate(sat_sim_bands, start=1):
        data_dict[f"SAT_sim_band{band_idx}"] = sat_band.flatten()
    
    # 添加 Sentinel-2 波段数据
    for band_idx, s2_band in enumerate(s2_bands, start=1):
        # 确保尺寸匹配
        if s2_band.shape != (sat_height, sat_width):
            s2_band = s2_band[:sat_height, :sat_width]
        data_dict[f"S2_band{band_idx}"] = s2_band.flatten()
    
    # 构建 DataFrame
    df = pd.DataFrame(data_dict)
    
    # ✨ 修复：过滤掉包含 nodata 值的行（同时检查 UAV 和 S2）
    initial_count = len(df)
    
    # 获取列名
    sat_sim_cols = [col for col in df.columns if col.startswith("SAT_sim_band")]
    s2_cols = [col for col in df.columns if col.startswith("S2_band")]
    
    # 创建过滤掩码
    mask = pd.Series(True, index=df.index)
    
    # 过滤仿卫星波段的 nodata 值
    if sat_nodata is not None:
        print(f"\n🔍 过滤 UAV 仿卫星 nodata 值: {sat_nodata}")
        for col in sat_sim_cols:
            mask = mask & (df[col] != sat_nodata) & (~pd.isna(df[col]))
    
    # ✨ 新增：过滤 Sentinel-2 波段的 nodata 值
    if s2_nodata is not None:
        print(f"🔍 过滤 Sentinel-2 nodata 值: {s2_nodata}")
        for col in s2_cols:
            mask = mask & (df[col] != s2_nodata) & (~pd.isna(df[col]))
    
    # 应用过滤
    df = df[mask].copy()
    filtered_count = initial_count - len(df)
    
    if filtered_count > 0:
        print(f"\n⚠️ 已过滤掉 {filtered_count} 行包含 nodata 值的数据")
        print(f"   - 过滤前: {initial_count} 行")
        print(f"   - 过滤后: {len(df)} 行")
        print(f"   - 过滤率: {filtered_count / initial_count * 100:.2f}%")
    
    print("\n✅ 配对数据构建完成：")
    print(f"   - 总像元数（过滤前）: {initial_count}")
    print(f"   - 总像元数（过滤后）: {len(df)}")
    print(f"   - 列数: {len(df.columns)}")
    print(f"   - 仿卫星波段数: {len(sat_sim_bands)}")
    print(f"   - Sentinel-2 波段数: {len(s2_bands)}")
    
    return df


def save_satellite_simulation(
    sat_sim_bands: list[np.ndarray],
    output_path: Path,
    transform: Affine,
    crs: rasterio.crs.CRS,
    dtype: str = "float32",
    nodata: float | None = None,
) -> None:
    """
    保存聚合后的仿卫星像元为多波段 GeoTIFF
    
    参数:
        sat_sim_bands: list，包含各波段聚合后的 numpy 数组
        output_path: Path，输出 GeoTIFF 文件路径
        transform: Affine，聚合后的 transform
        crs: rasterio.crs.CRS，坐标参考系统（使用原 UAV CRS）
        dtype: str，输出数据类型，默认为 "float32"
        nodata: float | None，nodata 值，如果为 None 则自动设置
    """
    if not sat_sim_bands:
        raise ValueError("sat_sim_bands 不能为空")
    
    # 获取输出尺寸（使用第一个波段）
    height, width = sat_sim_bands[0].shape
    num_bands = len(sat_sim_bands)
    
    # 检查所有波段尺寸是否一致
    for i, band in enumerate(sat_sim_bands):
        if band.shape != (height, width):
            raise ValueError(
                f"波段 {i + 1} 的尺寸 {band.shape} 与其他波段不一致 ({height}, {width})"
            )
    
    # 自动设置 nodata 值
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
    
    # 写入 GeoTIFF
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
        for band_idx, band_array in enumerate(sat_sim_bands, start=1):
            # 处理 NaN 值
            if np.issubdtype(band_array.dtype, np.integer):
                band_array = band_array.astype(np.float32)
            
            # 将 NaN 值转换为 nodata 值
            if np.isnan(band_array).any():
                band_array = np.where(np.isnan(band_array), nodata, band_array)
                print(f"   波段 {band_idx}: 已将 NaN 值转换为 nodata ({nodata})")
            
            # 转换到目标数据类型
            if dtype != str(band_array.dtype):
                band_array = band_array.astype(dtype)
            
            dst.write(band_array, band_idx)
            print(f"✅ 波段 {band_idx}/{num_bands} 已写入")
    
    print(f"\n✅ 仿卫星像元 GeoTIFF 已保存：{output_path}")
    print(f"   - 波段数: {num_bands}")
    print(f"   - 尺寸: {height} × {width}")
    print(f"   - 数据类型: {dtype}")
    print(f"   - Nodata: {nodata}")


# ===============================================================================
#                          主程序入口
# ===============================================================================

def main():
    """主处理流程"""
    
    print("\n" + "=" * 80)
    print("UAV 影像聚合生成仿卫星像元处理流程")
    print("=" * 80)
    
    # -----------------------------------------------------------------------
    # 步骤 1: 读取 UAV 影像
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("步骤 1/5: 读取 UAV 影像")
    print("=" * 60)
    
    with rasterio.open(UAV_RASTER_PATH) as src:
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
            print(f"✅ 波段 {band_idx}: shape = {band_data.shape}, dtype = {band_data.dtype}")
    
    print(f"\n📊 UAV 影像信息:")
    print(f"   - 影像路径: {UAV_RASTER_PATH}")
    print(f"   - 波段数: {num_bands}")
    print(f"   - 尺寸: {height} × {width}")
    print(f"   - 像元分辨率 X: {pixel_size_x}")
    print(f"   - 像元分辨率 Y: {pixel_size_y}")
    print(f"   - CRS: {crs}")
    
    # -----------------------------------------------------------------------
    # 步骤 2: 空间聚合
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("步骤 2/5: 空间聚合到卫星分辨率")
    print("=" * 60)
    print(f"目标分辨率: {TARGET_RESOLUTION}")
    
    target_resolutions = [TARGET_RESOLUTION] * num_bands
    
    sat_sim_bands, new_transform = aggregate_to_satellite_resolution(
        uav_band_arrays=bands,
        uav_transform=transform,
        uav_pixel_size_x=pixel_size_x,
        uav_pixel_size_y=pixel_size_y,
        target_resolution=target_resolutions,
        nodata_value=UAV_NODATA,
    )
    
    # -----------------------------------------------------------------------
    # 步骤 3: 对齐到 Sentinel-2 网格
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("步骤 3/5: 对齐到 Sentinel-2 像元网格")
    print("=" * 60)
    
    aligned_bands, s2_aligned_transform, output_shape = align_to_sentinel2_grid(
        uav_aggregated_bands=sat_sim_bands,
        uav_transform=new_transform,
        uav_crs=crs,
        sentinel2_path=SENTINEL2_RASTER_PATH,
    )
    
    # -----------------------------------------------------------------------
    # 步骤 4: 保存仿卫星像元 GeoTIFF
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("步骤 4/5: 保存对齐后的仿卫星像元 GeoTIFF")
    print("=" * 60)
    
    with rasterio.open(SENTINEL2_RASTER_PATH) as s2_src:
        s2_crs = s2_src.crs
    
    save_satellite_simulation(
        sat_sim_bands=aligned_bands,
        output_path=OUTPUT_ALIGNED_TIFF,
        transform=s2_aligned_transform,
        crs=s2_crs,
        dtype="float32",
        nodata=None,
    )
    
    # -----------------------------------------------------------------------
    # 步骤 5: 提取像元配对数据
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("步骤 5/5: 提取像元配对数据")
    print("=" * 60)
    
    pixel_pairs_df = extract_pixel_pairs(
        sat_sim_bands=aligned_bands,
        sat_sim_transform=s2_aligned_transform,
        sentinel2_path=SENTINEL2_RASTER_PATH,
        sat_nodata=UAV_NODATA,
        s2_nodata=S2_NODATA,
    )
    
    # 保存配对数据为 CSV
    OUTPUT_PIXEL_PAIRS_CSV.parent.mkdir(parents=True, exist_ok=True)
    pixel_pairs_df.to_csv(OUTPUT_PIXEL_PAIRS_CSV, index=False, encoding="utf-8-sig")
    
    print(f"\n✅ 配对数据已保存：{OUTPUT_PIXEL_PAIRS_CSV}")
    print(f"   - 数据行数: {len(pixel_pairs_df)}")
    print(f"   - 数据列数: {len(pixel_pairs_df.columns)}")
    
    # -----------------------------------------------------------------------
    # 完成
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("✅ 全部处理完成！")
    print("=" * 80)
    print(f"\n输出文件:")
    print(f"   1. 仿卫星像元影像: {OUTPUT_ALIGNED_TIFF}")
    print(f"   2. 像元配对数据: {OUTPUT_PIXEL_PAIRS_CSV}")
    print()


if __name__ == "__main__":
    main()