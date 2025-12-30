"""
================================================================================
代码功能：从样点位置提取多源遥感影像的光谱值
================================================================================

【核心目的】
从给定的样点位置（shapefile格式）提取两种影像的光谱值：
1. Sentinel-2 原始卫星影像的光谱值
2. 通过模型转换后的"仿卫星像元"影像的光谱值

这个代码是星机光谱融合反演工作流程中的验证环节，用于评估光谱转换模型的效果。

【应用场景】
假设你已经：
1. ✅ 用 UAV 影像聚合生成了"仿卫星像元"
2. ✅ 训练了一个光谱映射模型（UAV → Sentinel-2）
3. ✅ 用模型将"仿卫星像元"转换成了"预测的S2光谱"

现在需要：
- 在验证样点位置提取"真实 S2 光谱"和"预测 S2 光谱"
- 对比两者的差异，评估模型精度

【处理流程】

输入数据：
├── 样点 Shapefile (138个样点)
├── Sentinel-2 原始影像（真实卫星数据）
└── 转换后的仿卫星像元影像（模型预测结果）

处理步骤：
1. 读取样点 Shapefile → 获取样点坐标
2. 在 S2 影像中提取样点位置的像元值 → S2_band1, S2_band2, ...
3. 在转换影像中提取样点位置的像元值 → SAT_sim_band1, SAT_sim_band2, ...
4. 合并数据 → 每个样点一行，包含真实值和预测值
5. 保存为 CSV → 用于后续统计分析和精度评估
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask

# ================= 用户输入区域 =================
samples_shapefile = Path(
    r"D:\富平星机光谱融合反演\数据\samples138_.shp"
)
s2_raster_path = Path(
    r"D:\富平星机光谱融合反演\数据\Fuping_Sentinel2_20240830.tif"
)
converted_satellite_sim_raster_path = Path(
    r"D:\富平星机光谱融合反演\结果\S2_converted_to_satellite_sim_v2.tif"
)
output_csv = Path(
    r"D:\富平星机光谱融合反演\middata\S2_samples_extracted_values_v2.csv"
)
# =================================================


def extract_raster_values_at_points(
    raster_path: Path,
    gdf: gpd.GeoDataFrame,
    band_names_prefix: str = "band",
) -> pd.DataFrame:
    """
    从栅格影像中提取样点位置的像元值
    
    参数:
        raster_path: 栅格影像路径
        gdf: GeoDataFrame，包含样点几何信息
        band_names_prefix: 波段列名前缀
    
    返回:
        DataFrame，包含提取的像元值
    """
    print(f"\n📂 读取栅格影像：{raster_path}")
    
    if not raster_path.exists():
        raise FileNotFoundError(f"未找到栅格文件：{raster_path}")
    
    with rasterio.open(raster_path) as src:
        num_bands = src.count
        crs = src.crs
        transform = src.transform
        
        print(f"   波段数: {num_bands}")
        print(f"   CRS: {crs}")
        print(f"   尺寸: {src.height} × {src.width}")
        
        # 检查 CRS 是否一致
        if gdf.crs != crs:
            print(f"   ⚠️ 样点 CRS ({gdf.crs}) 与影像 CRS ({crs}) 不一致，将重投影样点...")
            gdf_reprojected = gdf.to_crs(crs)
        else:
            gdf_reprojected = gdf.copy()
        
        # 提取样点位置的像元值
        extracted_values = []
        
        for idx, row in gdf_reprojected.iterrows():
            geom = row.geometry
            
            # 使用 sample 方法提取像元值
            sample_values = []
            for band_idx in range(1, num_bands + 1):
                # sample 返回一个生成器，需要转换为列表
                values = list(src.sample([(geom.x, geom.y)], indexes=[band_idx]))
                if values:
                    sample_values.append(values[0][0])
                else:
                    sample_values.append(np.nan)
            
            extracted_values.append(sample_values)
        
        # 创建 DataFrame
        band_columns = [f"{band_names_prefix}{i}" for i in range(1, num_bands + 1)]
        values_df = pd.DataFrame(extracted_values, columns=band_columns)
        
        print(f"✅ 成功提取 {len(values_df)} 个样点的像元值")
        print(f"   提取的波段列: {band_columns}")
    
    return values_df


def main() -> None:
    print("=" * 60)
    print("从样点提取影像像元值")
    print("=" * 60)
    
    # 1. 读取样点 shapefile
    print(f"\n📂 读取样点 shapefile：{samples_shapefile}")
    if not samples_shapefile.exists():
        raise FileNotFoundError(f"未找到样点文件：{samples_shapefile}")
    
    gdf = gpd.read_file(samples_shapefile)
    print(f"✅ 样点读取完成：{len(gdf)} 个样点")
    print(f"   CRS: {gdf.crs}")
    print(f"   列名: {list(gdf.columns)}")
    
    # 显示样点的前几行信息
    print(f"\n样点信息预览：")
    print(gdf.head())
    
    # 2. 提取 S2 影像值
    print("\n" + "=" * 60)
    print("提取 S2 影像像元值")
    print("=" * 60)
    
    s2_values = extract_raster_values_at_points(
        raster_path=s2_raster_path,
        gdf=gdf,
        band_names_prefix="S2_band",
    )
    
    # 3. 提取转换后仿卫星像元影像值
    print("\n" + "=" * 60)
    print("提取转换后仿卫星像元影像值")
    print("=" * 60)
    
    satellite_sim_values = extract_raster_values_at_points(
        raster_path=converted_satellite_sim_raster_path,
        gdf=gdf,
        band_names_prefix="SAT_sim_band",
    )
    
    # 4. 合并数据
    print("\n📊 合并数据...")
    
    # 合并样点属性和提取的值
    result_df = pd.concat([
        gdf.drop(columns=['geometry']),  # 移除几何列（可选，如果需要保留坐标可以添加）
        s2_values,
        satellite_sim_values,
    ], axis=1)
    
    # 如果原始 shapefile 有坐标列，也可以添加
    if 'geometry' in gdf.columns:
        result_df['longitude'] = gdf.geometry.x
        result_df['latitude'] = gdf.geometry.y
    
    print(f"✅ 数据合并完成")
    print(f"   总列数: {len(result_df.columns)}")
    print(f"   总行数: {len(result_df)}")
    
    # 显示统计信息
    print(f"\n📊 数据统计：")
    print(f"   S2 波段数: {len(s2_values.columns)}")
    print(f"   仿卫星像元波段数: {len(satellite_sim_values.columns)}")
    
    # 检查缺失值
    missing_s2 = s2_values.isnull().sum().sum()
    missing_sat = satellite_sim_values.isnull().sum().sum()
    print(f"   S2 缺失值数量: {missing_s2}")
    print(f"   仿卫星像元缺失值数量: {missing_sat}")
    
    # 5. 保存结果
    print("\n" + "=" * 60)
    print("保存结果")
    print("=" * 60)
    
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    
    print(f"✅ 结果已保存到：{output_csv}")
    print(f"   总行数: {len(result_df)}")
    print(f"   总列数: {len(result_df.columns)}")
    
    # 显示前几行
    print(f"\n结果预览（前5行）：")
    print(result_df.head())
    
    print("\n✅ 处理完成！")


if __name__ == "__main__":
    main()

