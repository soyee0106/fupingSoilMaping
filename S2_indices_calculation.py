"""
根据无人机光谱特征 CSV 计算各类指数（植被、土壤等），
模仿 `indicesCalculation.py` 的公式体系。

使用方式：
1. 修改 `input_csv`, `output_csv`, `band_map` 以匹配实际文件与波段映射。
2. 运行 `python uav_indices_calculation.py`。
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import rasterio

# ================= 用户输入区域 =================
input_csv = Path(
    r"D:\富平星机光谱融合反演\middata\S2_samples_extracted_values_v2.csv"
)
output_csv = Path(
    r"D:\富平星机光谱融合反演\middata\Sim_samples_extracted_values.csv_with_indices_v2.csv"
)
# 将 UAV 波段列映射到公式变量，按实际波段顺序调整
# band_map: Dict[str, str] = {
#     "S2_band3": "G",  # Green
#     "S2_band4": "R",  # Red
#     "S2_band6": "REG",  # Red Edge
#     "S2_band9": "NIR",  # Near Infrared
# }
band_map: Dict[str, str] = {
    "SAT_sim_band1": "G",  # Green
    "SAT_sim_band2": "R",  # Red
    "SAT_sim_band3": "REG",  # Red Edge
    "SAT_sim_band4": "NIR",  # Near Infrared
}
L = 0.5  # SAVI 中的土壤调节系数

# 栅格影像处理参数
input_raster_path = Path(
    r"D:\富平星机光谱融合反演\数据\Fuping_Sentinel2_20240830.tif"
)  # 输入栅格影像路径
output_feature_raster_path = Path(
    r"D:\富平星机光谱融合反演\数据\S2_features_30bands.tif"
)  # 输出的30个特征波段栅格路径
# 波段映射：栅格波段索引（从0开始）→ 波段名称
raster_band_map: Dict[int, str] = {
    0: "G",   # 波段1 → Green (对应 SAT_sim_band1 或 S2_band3)
    1: "R",   # 波段2 → Red (对应 SAT_sim_band2 或 S2_band4)
    2: "REG", # 波段3 → Red Edge (对应 SAT_sim_band3 或 S2_band6)
    3: "NIR", # 波段4 → Near Infrared (对应 SAT_sim_band4 或 S2_band9)
}
# =================================================


def safe_div_array(
    numerator: np.ndarray, denominator: np.ndarray
) -> np.ndarray:
    """数组除法安全处理，避免除以 0。"""
    denominator = np.where(denominator == 0, np.nan, denominator)
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator != 0)


def safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """除法安全处理，避免除以 0。"""
    denom = denominator.replace(0, np.nan)
    return numerator / denom


def calculate_indices_from_arrays(
    G: np.ndarray, R: np.ndarray, REG: np.ndarray, NIR: np.ndarray, L: float = 0.5
) -> Dict[str, np.ndarray]:
    """
    从4个波段数组计算所有指数，返回字典
    
    参数:
        G: Green 波段数组
        R: Red 波段数组
        REG: Red Edge 波段数组
        NIR: Near Infrared 波段数组
        L: SAVI 土壤调节系数
    
    返回:
        包含所有指数数组的字典
    """
    indices = {}
    
    # S1, S1REG
    indices["S1"] = safe_div_array(R * NIR, G)
    indices["S1REG"] = safe_div_array(REG * NIR, G)
    
    # NDSI, NDSIREG
    indices["NDSI"] = safe_div_array(R - NIR, R + NIR)
    indices["NDSIREG"] = safe_div_array(REG - NIR, REG + NIR)
    
    # SI1, SI1REG
    indices["SI1"] = np.sqrt(G * R)
    indices["SI1REG"] = np.sqrt(G * REG)
    
    # SI2, SI2REG
    indices["SI2"] = np.sqrt(G ** 2 + R ** 2 + NIR ** 2)
    indices["SI2REG"] = np.sqrt(G ** 2 + REG ** 2 + NIR ** 2)
    
    # SI3, SI3REG
    indices["SI3"] = np.sqrt(G ** 2 + R ** 2)
    indices["SI3REG"] = np.sqrt(G ** 2 + REG ** 2)
    
    # SIT, SITREG
    indices["SIT"] = 100 * (R - NIR)
    indices["SITREG"] = 100 * (REG - NIR)
    
    # Int1, Int1REG
    indices["Int1"] = (G + R) / 2
    indices["Int1REG"] = (G + REG) / 2
    
    # Int2, Int2REG
    indices["Int2"] = (G + R + NIR) / 2
    indices["Int2REG"] = (G + REG + NIR) / 2
    
    # NDVI, NDVIREG
    indices["NDVI"] = safe_div_array(NIR - R, NIR + R)
    indices["NDVIREG"] = safe_div_array(NIR - REG, NIR + REG)
    
    # SAVI, SAVIREG
    indices["SAVI"] = safe_div_array(
        (1 + L) * (NIR - R), NIR + R + L
    )
    indices["SAVIREG"] = safe_div_array(
        (1 + L) * (NIR - REG), NIR + REG + L
    )
    
    # BI, BIREG
    indices["BI"] = np.sqrt(R ** 2 + NIR ** 2)
    indices["BIREG"] = np.sqrt(REG ** 2 + NIR ** 2)
    
    # IFe2O3, IFe2O3REG
    indices["IFe2O3"] = safe_div_array(R, NIR)
    indices["IFe2O3REG"] = safe_div_array(REG, NIR)
    
    # DVI, DVIREG
    indices["DVI"] = NIR - R
    indices["DVIREG"] = NIR - REG
    
    return indices


def extract_30_feature_bands_from_raster(
    raster_path: Path,
    output_path: Path,
    band_map: Dict[int, str],
    L: float = 0.5,
) -> Tuple[List[np.ndarray], List[str]]:
    """
    从栅格影像中提取30个特征波段（4个原始波段 + 26个指数）
    
    参数:
        raster_path: 输入栅格影像路径
        output_path: 输出的30波段特征栅格路径
        band_map: 波段索引到波段名称的映射 {band_index: band_name}
        L: SAVI 土壤调节系数
    
    返回:
        (feature_bands, band_names): 特征波段数组列表和波段名称列表
    """
    print(f"\n📂 读取栅格影像：{raster_path}")
    
    if not raster_path.exists():
        raise FileNotFoundError(f"未找到栅格文件：{raster_path}")
    
    with rasterio.open(raster_path) as src:
        height = src.height
        width = src.width
        count = src.count
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
        
        print(f"   尺寸: {height} × {width}")
        print(f"   波段数: {count}")
        print(f"   CRS: {crs}")
        
        # 读取所有波段数据
        all_bands = src.read()  # (bands, height, width)
        print("✅ 影像读取完成")
    
    # 提取4个原始波段
    print("\n📊 提取原始波段...")
    band_arrays = {}
    for band_idx, band_name in band_map.items():
        if band_idx >= count:
            raise ValueError(
                f"波段索引 {band_idx} 超出影像波段数 ({count})"
            )
        band_data = all_bands[band_idx, :, :].astype(np.float32)
        
        # 处理nodata值
        if nodata is not None:
            band_data = np.where(band_data == nodata, np.nan, band_data)
        
        band_arrays[band_name] = band_data
        valid_count = np.isfinite(band_data).sum()
        print(f"   波段 {band_idx + 1} ({band_name}): shape={band_data.shape}, "
              f"范围=[{np.nanmin(band_data):.6f}, {np.nanmax(band_data):.6f}], "
              f"有效像素={valid_count}/{band_data.size}")
    
    # 获取4个波段数组
    G = band_arrays.get("G")
    R = band_arrays.get("R")
    REG = band_arrays.get("REG")
    NIR = band_arrays.get("NIR")
    
    if G is None or R is None or REG is None or NIR is None:
        missing = [k for k in ["G", "R", "REG", "NIR"] if k not in band_arrays]
        raise ValueError(f"缺少必要的波段：{missing}")
    
    # 计算所有指数
    print("\n📊 计算指数...")
    indices = calculate_indices_from_arrays(G, R, REG, NIR, L)
    print(f"✅ 成功计算 {len(indices)} 个指数")
    
    # 构建30个特征波段列表（按顺序）
    # 顺序：4个原始波段 + 26个指数（按计算顺序）
    feature_bands = []
    band_names = []
    
    # 1. 原始波段（按顺序：G, R, REG, NIR）
    feature_bands.append(G)
    band_names.append("G")
    feature_bands.append(R)
    band_names.append("R")
    feature_bands.append(REG)
    band_names.append("REG")
    feature_bands.append(NIR)
    band_names.append("NIR")
    
    # 2. 指数（按计算顺序）
    index_order = [
        "S1", "S1REG",
        "NDSI", "NDSIREG",
        "SI1", "SI1REG",
        "SI2", "SI2REG",
        "SI3", "SI3REG",
        "SIT", "SITREG",
        "Int1", "Int1REG",
        "Int2", "Int2REG",
        "NDVI", "NDVIREG",
        "SAVI", "SAVIREG",
        "BI", "BIREG",
        "IFe2O3", "IFe2O3REG",
        "DVI", "DVIREG",
    ]
    
    for idx_name in index_order:
        if idx_name in indices:
            feature_bands.append(indices[idx_name])
            band_names.append(idx_name)
        else:
            print(f"⚠️ 警告：指数 {idx_name} 未计算，跳过")
    
    if len(feature_bands) != 30:
        print(f"⚠️ 警告：特征波段数 ({len(feature_bands)}) 不等于30，继续保存...")
    
    print(f"\n✅ 特征波段准备完成：共 {len(feature_bands)} 个波段")
    
    # 保存为多波段 GeoTIFF
    print("\n💾 保存特征波段栅格...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    nodata_value = -9999.0
    
    # 将NaN替换为nodata值
    feature_bands_clean = []
    for band in feature_bands:
        band_clean = np.where(np.isfinite(band), band, nodata_value)
        feature_bands_clean.append(band_clean)
    
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=len(feature_bands_clean),
        dtype=np.float32,
        crs=crs,
        transform=transform,
        nodata=nodata_value,
        compress='lzw',
    ) as dst:
        for band_idx, (band_data, band_name) in enumerate(zip(feature_bands_clean, band_names), start=1):
            dst.write(band_data, band_idx)
            
            # 计算统计信息
            valid_mask = band_data != nodata_value
            if valid_mask.any():
                valid_data = band_data[valid_mask]
                print(f"   波段 {band_idx:2d}/{len(feature_bands_clean)}: {band_name:10s} "
                      f"范围=[{valid_data.min():8.4f}, {valid_data.max():8.4f}] "
                      f"均值={valid_data.mean():8.4f}")
            else:
                print(f"   波段 {band_idx:2d}/{len(feature_bands_clean)}: {band_name:10s} (无有效像素)")
    
    print(f"\n✅ 特征波段栅格已保存：{output_path}")
    print(f"   波段数: {len(feature_bands_clean)}")
    print(f"   波段名称: {', '.join(band_names)}")
    
    return feature_bands_clean, band_names


def main() -> None:
    if not input_csv.exists():
        raise FileNotFoundError(f"未找到输入 CSV：{input_csv}")

    df = pd.read_csv(input_csv)
    if df.empty:
        raise ValueError("输入 CSV 为空，无法计算指数。")

    # 将需要的波段复制到公式变量
    for column, alias in band_map.items():
        if column not in df.columns:
            raise KeyError(f"列 {column} 不存在，请检查无人机特征 CSV 或调整 band_map。")
        df[alias] = df[column]

    # 计算各类指数（与 indicesCalculation.py 一致）
    df["S1"] = safe_div(df["R"] * df["NIR"], df["G"])
    df["S1REG"] = safe_div(df["REG"] * df["NIR"], df["G"])
    df["NDSI"] = safe_div(df["R"] - df["NIR"], df["R"] + df["NIR"])
    df["NDSIREG"] = safe_div(df["REG"] - df["NIR"], df["REG"] + df["NIR"])
    df["SI1"] = np.sqrt(df["G"] * df["R"])
    df["SI1REG"] = np.sqrt(df["G"] * df["REG"])
    df["SI2"] = np.sqrt(df["G"] ** 2 + df["R"] ** 2 + df["NIR"] ** 2)
    df["SI2REG"] = np.sqrt(df["G"] ** 2 + df["REG"] ** 2 + df["NIR"] ** 2)
    df["SI3"] = np.sqrt(df["G"] ** 2 + df["R"] ** 2)
    df["SI3REG"] = np.sqrt(df["G"] ** 2 + df["REG"] ** 2)
    df["SIT"] = 100 * (df["R"] - df["NIR"])
    df["SITREG"] = 100 * (df["REG"] - df["NIR"])

    df["Int1"] = (df["G"] + df["R"]) / 2
    df["Int1REG"] = (df["G"] + df["REG"]) / 2
    df["Int2"] = (df["G"] + df["R"] + df["NIR"]) / 2
    df["Int2REG"] = (df["G"] + df["REG"] + df["NIR"]) / 2

    df["NDVI"] = safe_div(df["NIR"] - df["R"], df["NIR"] + df["R"])
    df["NDVIREG"] = safe_div(df["NIR"] - df["REG"], df["NIR"] + df["REG"])
    df["SAVI"] = safe_div((1 + L) * (df["NIR"] - df["R"]), df["NIR"] + df["R"] + L)
    df["SAVIREG"] = safe_div((1 + L) * (df["NIR"] - df["REG"]), df["NIR"] + df["REG"] + L)

    df["BI"] = np.sqrt(df["R"] ** 2 + df["NIR"] ** 2)
    df["BIREG"] = np.sqrt(df["REG"] ** 2 + df["NIR"] ** 2)
    df["IFe2O3"] = safe_div(df["R"], df["NIR"])
    df["IFe2O3REG"] = safe_div(df["REG"], df["NIR"])

    df["DVI"] = df["NIR"] - df["R"]
    df["DVIREG"] = df["NIR"] - df["REG"]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"🎯 指数计算完成：{output_csv}")


def main_raster() -> None:
    """从栅格影像提取30个特征波段"""
    if not input_raster_path.exists():
        raise FileNotFoundError(f"未找到输入栅格：{input_raster_path}")
    
    extract_30_feature_bands_from_raster(
        raster_path=input_raster_path,
        output_path=output_feature_raster_path,
        band_map=raster_band_map,
        L=L,
    )


if __name__ == "__main__":
    # 选择运行模式
    # main()  # 从CSV计算指数
    main_raster()  # 从栅格影像提取30个特征波段

