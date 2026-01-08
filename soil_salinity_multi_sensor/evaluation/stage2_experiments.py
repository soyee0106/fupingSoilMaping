"""
Stage 2 盐分反演实验脚本

参照 S2_salinity_inversion_plsr_vip.py 的逻辑，实现4组对比实验：
1. 原始UAV影像反演盐分（30个FP样点）
2. 原始S2影像反演盐分（108个非FP样点）
3. 原始L8影像反演盐分（108个非FP样点）
4. 校正后的UAV光谱反演盐分（108个非FP样点）

所有实验使用相同的模型和特征（30个特征：4波段+26指数）
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import rowcol
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from scipy.stats import pearsonr, spearmanr
import yaml
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from data_preprocessing.spectral_indices import calculate_all_indices

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置中文字体
plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ================= 配置参数 =================
# 从配置文件读取路径
CONFIG_PATH = "configs/data_config.yaml"
OUTPUT_BASE_DIR = Path("outputs/stage2_experiments")

# 实验配置
EXPERIMENTS = {
    "exp1_uav_original": {
        "name": "实验1：原始UAV影像反演",
        "raster_path": None,  # 从config读取
        "sample_filter": {"column": "抽样编", "prefix": "FP"},  # FP开头的30个点
        "band_mapping": {0: "G", 1: "R", 2: "REG", 3: "NIR"},  # UAV波段顺序
        "output_dir": OUTPUT_BASE_DIR / "exp1_uav_original"
    },
    "exp2_s2_original": {
        "name": "实验2：原始S2影像反演",
        "raster_path": None,  # 从config读取
        "sample_filter": {"column": "抽样编", "exclude_prefix": "FP"},  # 非FP开头的108个点
        "band_mapping": {2: "G", 3: "R", 4: "REG", 7: "NIR"},  # S2波段索引（从0开始，实际是3,4,5,8）
        "output_dir": OUTPUT_BASE_DIR / "exp2_s2_original",
        "use_band_mapping_config": True  # 使用配置文件中的波段映射
    },
    "exp3_l8_original": {
        "name": "实验3：原始L8影像反演",
        "raster_path": None,  # 从config读取
        "sample_filter": {"column": "抽样编", "exclude_prefix": "FP"},  # 非FP开头的108个点
        "band_mapping": {2: "G", 3: "R", None: "REG", 4: "NIR"},  # L8波段索引（REG缺失，用0填充）
        "output_dir": OUTPUT_BASE_DIR / "exp3_l8_original",
        "use_band_mapping_config": True  # 使用配置文件中的波段映射
    },
    "exp4_s2_corrected": {
        "name": "实验4：S2校正后UAV光谱反演",
        "raster_path": Path("outputs/stage1_inference/s2_corrected_uav_spectrum.tif"),
        "sample_filter": {"column": "抽样编", "exclude_prefix": "FP"},  # 非FP开头的108个点
        "band_mapping": {0: "G", 1: "R", 2: "REG", 3: "NIR"},  # 校正后的UAV波段顺序
        "output_dir": OUTPUT_BASE_DIR / "exp4_s2_corrected"
    },
    "exp5_l8_corrected": {
        "name": "实验5：L8校正后UAV光谱反演",
        "raster_path": Path("outputs/stage1_inference/l8_corrected_uav_spectrum.tif"),
        "sample_filter": {"column": "抽样编", "exclude_prefix": "FP"},  # 非FP开头的108个点
        "band_mapping": {0: "G", 1: "R", 2: "REG", 3: "NIR"},  # 校正后的UAV波段顺序
        "output_dir": OUTPUT_BASE_DIR / "exp5_l8_corrected"
    }
}

# 模型参数（参照S2_salinity_inversion_plsr_vip.py）
PEARSON_THRESHOLD = 0.3
RFE_N_FEATURES = 15
RFE_ESTIMATOR = "RandomForest"
CV_FOLDS = 5
TEST_RATIO = 0.2
RANDOM_STATE = 42

SVR_PARAM_GRID = {
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
    "epsilon": [0.01, 0.1, 0.5],
}


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def extract_raster_values_at_points(
    raster_path: Path,
    points_gdf: gpd.GeoDataFrame,
    band_indices: list,
    band_names: list,
    band_mapping_config: dict = None
) -> pd.DataFrame:
    """
    从栅格影像中提取样点位置的波段值
    
    参数:
        raster_path: 栅格影像路径
        points_gdf: 样点GeoDataFrame
        band_indices: 波段索引列表（从0开始）
        band_names: 波段名称列表
    
    返回:
        DataFrame，包含提取的波段值
    """
    logger.info(f"从栅格提取样点值: {raster_path}")
    
    if not raster_path.exists():
        raise FileNotFoundError(f"未找到栅格文件: {raster_path}")
    
    with rasterio.open(raster_path) as src:
        # 确保CRS一致
        if points_gdf.crs != src.crs:
            logger.warning(f"CRS不一致，将样点从 {points_gdf.crs} 转换为 {src.crs}")
            points_gdf = points_gdf.to_crs(src.crs)
        
        # 如果提供了band_mapping_config，使用配置查找波段索引
        if band_mapping_config is not None:
            from data_preprocessing.data_pairing import DataPairer
            pairer = DataPairer()
            found_bands = pairer.find_bands_by_mapping(raster_path, band_mapping_config, ['G', 'R', 'REG', 'NIR'])
            # 更新band_indices
            band_indices = []
            for band_name in band_names:
                if band_name in found_bands:
                    idx = found_bands[band_name]
                    if idx is not None:
                        band_indices.append(idx - 1)  # 转换为从0开始
                    else:
                        band_indices.append(None)
                else:
                    band_indices.append(None)
            logger.info(f"使用配置查找波段: {found_bands}")
        
        # 提取样点坐标
        points_coords = [(geom.x, geom.y) for geom in points_gdf.geometry]
        
        # 提取每个波段的值
        band_data = {}
        for band_idx, band_name in zip(band_indices, band_names):
            if band_idx is None:
                # 缺失波段（如L8的REG），填充为0
                band_data[band_name] = np.zeros(len(points_gdf), dtype=np.float32)
                logger.warning(f"波段 {band_name} 缺失，填充为0")
            else:
                # 实际波段索引（rasterio从1开始，但band_idx已经是0-based）
                values = []
                for x, y in points_coords:
                    try:
                        row, col = rowcol(src.transform, x, y)
                        if 0 <= row < src.height and 0 <= col < src.width:
                            val = src.read(band_idx + 1, window=rasterio.windows.Window(col, row, 1, 1))
                            values.append(float(val[0, 0]))
                        else:
                            values.append(np.nan)
                    except Exception as e:
                        logger.warning(f"提取样点 ({x}, {y}) 的值时出错: {e}")
                        values.append(np.nan)
                band_data[band_name] = np.array(values, dtype=np.float32)
        
        # 创建DataFrame
        result_df = pd.DataFrame(band_data)
        
        logger.info(f"成功提取 {len(result_df)} 个样点的波段值")
        logger.info(f"有效值数量: {result_df.notna().all(axis=1).sum()}")
        
        return result_df


def calculate_features_from_bands(
    bands_df: pd.DataFrame,
    band_names: list = ["G", "R", "REG", "NIR"]
) -> pd.DataFrame:
    """
    从波段值计算30个特征（4个波段 + 26个指数）
    
    参数:
        bands_df: 包含波段值的DataFrame
        band_names: 波段名称列表
    
    返回:
        DataFrame，包含30个特征
    """
    logger.info("计算光谱指数...")
    
    # 准备波段字典
    bands_dict = {}
    for band_name in band_names:
        if band_name in bands_df.columns:
            bands_dict[band_name] = bands_df[band_name].values
        else:
            raise ValueError(f"缺少波段: {band_name}")
    
    # 计算所有指数
    indices = calculate_all_indices(bands_dict, L=0.5)
    
    # 合并波段和指数
    features_df = bands_df.copy()
    for idx_name, idx_values in indices.items():
        features_df[f"Index_{idx_name}"] = idx_values
    
    # 重命名列，统一格式
    feature_cols = []
    for band_name in band_names:
        feature_cols.append(f"Band_{band_name}")
    for idx_name in sorted(indices.keys()):
        feature_cols.append(f"Index_{idx_name}")
    
    # 重命名DataFrame列
    rename_dict = {}
    for band_name in band_names:
        rename_dict[band_name] = f"Band_{band_name}"
    features_df = features_df.rename(columns=rename_dict)
    
    # 确保列顺序正确
    features_df = features_df[feature_cols]
    
    logger.info(f"特征计算完成: {features_df.shape[1]} 个特征")
    
    return features_df


def select_features_by_pearson(
    X: np.ndarray, y: np.ndarray, feature_names: list, threshold: float = 0.3
) -> tuple:
    """基于皮尔逊相关系数选择特征"""
    logger.info(f"皮尔逊相关性特征选择（阈值={threshold}）...")
    
    correlations = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        corr, _ = pearsonr(X[:, i], y)
        correlations[i] = corr
    
    abs_correlations = np.abs(correlations)
    selected_mask = abs_correlations >= threshold
    selected_indices = np.where(selected_mask)[0]
    
    if len(selected_indices) < 5:
        logger.warning(f"阈值筛选后特征数过少（{len(selected_indices)}个），改为选择前15个")
        top_n = min(15, X.shape[1])
        selected_indices = np.argsort(abs_correlations)[::-1][:top_n]
    
    selected_features = [feature_names[i] for i in selected_indices]
    logger.info(f"皮尔逊相关性选择: {len(selected_features)} 个特征")
    
    return selected_features, selected_indices, correlations


def select_features_by_rfe(
    X: np.ndarray, y: np.ndarray, feature_names: list, n_features: int = 15,
    estimator_type: str = "RandomForest"
) -> tuple:
    """基于RFE选择特征"""
    logger.info(f"RFE特征选择（目标特征数={n_features}，估计器={estimator_type}）...")
    
    if estimator_type == "RandomForest":
        estimator = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    elif estimator_type == "SVR":
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        estimator = SVR(kernel="rbf", C=10, gamma="scale", epsilon=0.1)
    elif estimator_type == "PLSR":
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        estimator = PLSRegression(n_components=min(10, X.shape[1], X.shape[0]))
    else:
        raise ValueError(f"不支持的估计器类型: {estimator_type}")
    
    n_features = min(n_features, X.shape[0], X.shape[1])
    rfe = RFE(estimator=estimator, n_features_to_select=n_features, step=1)
    rfe.fit(X, y)
    
    selected_mask = rfe.support_
    selected_indices = np.where(selected_mask)[0]
    selected_features = [feature_names[i] for i in selected_indices]
    rfe_ranking = rfe.ranking_
    
    logger.info(f"RFE选择: {len(selected_features)} 个特征")
    
    return selected_features, selected_indices, rfe_ranking


def compare_and_select_features(
    X: np.ndarray, y: np.ndarray, feature_names: list
) -> tuple:
    """比较皮尔逊相关性和RFE，选择最终特征"""
    # 皮尔逊相关性
    pearson_features, pearson_indices, pearson_correlations = select_features_by_pearson(
        X, y, feature_names, threshold=PEARSON_THRESHOLD
    )
    
    # RFE
    rfe_features, rfe_indices, rfe_ranking = select_features_by_rfe(
        X, y, feature_names, n_features=RFE_N_FEATURES, estimator_type=RFE_ESTIMATOR
    )
    
    # 计算交集和并集
    pearson_set = set(pearson_features)
    rfe_set = set(rfe_features)
    intersection = pearson_set & rfe_set
    union = pearson_set | rfe_set
    
    # 选择策略：优先使用交集，如果交集太少则使用并集
    if len(intersection) >= 5:
        final_selected_features = list(intersection)
        selection_strategy = "交集"
    else:
        final_selected_features = list(union)
        selection_strategy = "并集"
    
    final_indices = [feature_names.index(f) for f in final_selected_features]
    
    selection_info = {
        "pearson_features": pearson_features,
        "rfe_features": rfe_features,
        "intersection": list(intersection),
        "union": list(union),
        "final_features": final_selected_features,
        "final_indices": final_indices,
        "selection_strategy": selection_strategy,
        "pearson_correlations": pearson_correlations,
        "rfe_ranking": rfe_ranking,
    }
    
    return final_selected_features, selection_info


def train_multiple_models(X_train, y_train, X_test, y_test, use_grid_search=True):
    """训练多个模型（参照S2_salinity_inversion_plsr_vip.py）"""
    results = {}
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 1. SVR
    logger.info("训练SVR模型...")
    if use_grid_search:
        svr = SVR(kernel="rbf")
        grid_search = GridSearchCV(
            svr, SVR_PARAM_GRID, cv=CV_FOLDS,
            scoring="neg_mean_squared_error", n_jobs=-1, verbose=0
        )
        grid_search.fit(X_train_scaled, y_train)
        svr_model = grid_search.best_estimator_
        logger.info(f"  最优参数: {grid_search.best_params_}")
    else:
        svr_model = SVR(kernel="rbf", C=10, gamma="scale", epsilon=0.1)
        svr_model.fit(X_train_scaled, y_train)
    
    y_pred_svr = svr_model.predict(X_test_scaled)
    results["SVR"] = {
        "model": svr_model,
        "scaler": scaler,
        "y_pred": y_pred_svr,
        "needs_scaling": True,
    }
    
    # 2. RandomForest
    logger.info("训练RandomForest模型...")
    rf_model = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    results["RandomForest"] = {
        "model": rf_model,
        "scaler": None,
        "y_pred": y_pred_rf,
        "needs_scaling": False,
    }
    
    # 3. PLSR
    logger.info("训练PLSR模型...")
    max_components = min(X_train_scaled.shape[0], X_train_scaled.shape[1])
    n_comp_range = range(1, min(21, max_components + 1))
    
    best_score = -np.inf
    optimal_n_comp = 1
    for n_comp in n_comp_range:
        plsr_temp = PLSRegression(n_components=n_comp)
        cv_scores = cross_val_score(
            plsr_temp, X_train_scaled, y_train, cv=CV_FOLDS,
            scoring="neg_mean_squared_error", n_jobs=-1
        )
        mean_score = cv_scores.mean()
        if mean_score > best_score:
            best_score = mean_score
            optimal_n_comp = n_comp
    
    logger.info(f"  最优主成分数: {optimal_n_comp}")
    plsr_model = PLSRegression(n_components=optimal_n_comp)
    plsr_model.fit(X_train_scaled, y_train)
    y_pred_plsr = plsr_model.predict(X_test_scaled)
    results["PLSR"] = {
        "model": plsr_model,
        "scaler": scaler,
        "y_pred": y_pred_plsr,
        "needs_scaling": True,
    }
    
    # 4. GradientBoosting
    logger.info("训练GradientBoosting模型...")
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE, max_depth=5)
    gb_model.fit(X_train, y_train)
    y_pred_gb = gb_model.predict(X_test)
    results["GradientBoosting"] = {
        "model": gb_model,
        "scaler": None,
        "y_pred": y_pred_gb,
        "needs_scaling": False,
    }
    
    # 5. XGBoost (如果可用)
    if HAS_XGBOOST:
        logger.info("训练XGBoost模型...")
        xgb_model = XGBRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
        xgb_model.fit(X_train, y_train)
        y_pred_xgb = xgb_model.predict(X_test)
        results["XGBoost"] = {
            "model": xgb_model,
            "scaler": None,
            "y_pred": y_pred_xgb,
            "needs_scaling": False,
        }
    
    return results


def calculate_metrics(y_true, y_pred, name, n_features):
    """计算评估指标"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
    r2 = r2_score(y_true, y_pred)
    adj_r2 = 1 - (1 - r2) * (len(y_true) - 1) / (len(y_true) - n_features - 1)
    evs = explained_variance_score(y_true, y_pred)
    pearson_r, _ = pearsonr(y_true, y_pred)
    spearman_r, _ = spearmanr(y_true, y_pred)
    
    return {
        "模型": name,
        "特征数": n_features,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MedAE": medae,
        "MAPE(%)": mape,
        "R²": r2,
        "Adj_R²": adj_r2,
        "EVS": evs,
        "Pearson_r": pearson_r,
        "Spearman_r": spearman_r,
    }


def plot_scatter(y_true, y_pred, title, save_path):
    """绘制散点图"""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolor="k")
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=2, label="1:1 线")
    plt.xlabel("真实全盐 (g/kg)")
    plt.ylabel("预测全盐 (g/kg)")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"散点图已保存: {save_path}")


def run_experiment(exp_config: dict, data_config: dict, samples_gdf: gpd.GeoDataFrame):
    """运行单个实验"""
    exp_name = exp_config["name"]
    output_dir = exp_config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info(f"运行实验: {exp_name}")
    logger.info("=" * 60)
    
    # 1. 过滤样点
    filter_config = exp_config["sample_filter"]
    filter_column = filter_config["column"]
    
    if "prefix" in filter_config:
        # 选择指定前缀的样点（实验1：FP开头）
        mask = samples_gdf[filter_column].astype(str).str.startswith(filter_config["prefix"])
        filtered_samples = samples_gdf[mask].copy()
        logger.info(f"筛选样点: {filter_config['prefix']}开头，共 {len(filtered_samples)} 个")
    elif "exclude_prefix" in filter_config:
        # 排除指定前缀的样点（实验2-4：非FP开头）
        mask = ~samples_gdf[filter_column].astype(str).str.startswith(filter_config["exclude_prefix"])
        filtered_samples = samples_gdf[mask].copy()
        logger.info(f"筛选样点: 排除{filter_config['exclude_prefix']}开头，共 {len(filtered_samples)} 个")
    else:
        filtered_samples = samples_gdf.copy()
    
    # 2. 确定栅格路径和波段映射
    if exp_config["raster_path"] is None:
        # 从config读取（实验1-3）
        if "uav" in exp_name.lower() and "corrected" not in exp_name.lower():
            raster_path = Path(data_config['data_paths']['uav_raster'])
            band_mapping_config = None
        elif "s2" in exp_name.lower() and "corrected" not in exp_name.lower():
            raster_path = Path(data_config['data_paths']['s2_raster'])
            band_mapping_config = data_config.get('bands', {}).get('satellite', {}).get('s2', {}).get('band_mapping', None)
        elif "l8" in exp_name.lower() and "corrected" not in exp_name.lower():
            raster_path = Path(data_config['data_paths']['l8_raster'])
            band_mapping_config = data_config.get('bands', {}).get('satellite', {}).get('l8', {}).get('band_mapping', None)
        else:
            raise ValueError(f"无法确定栅格路径: {exp_name}")
    else:
        # 直接使用配置的路径（实验4-5：校正后的UAV光谱）
        raster_path = exp_config["raster_path"]
        band_mapping_config = None
    
    raster_configs = [{
        "path": raster_path,
        "name": "raster",
        "band_mapping": exp_config["band_mapping"],
        "band_mapping_config": band_mapping_config if exp_config.get("use_band_mapping_config", False) else None
    }]
    
    # 3. 提取波段值
    raster_config = raster_configs[0]  # 每个实验只有一个栅格
    raster_path = raster_config["path"]
    
    if not raster_path.exists():
        raise FileNotFoundError(f"栅格文件不存在: {raster_path}")
    
    band_mapping = raster_config["band_mapping"]
    band_indices = []
    band_names = []
    for idx, name in band_mapping.items():
        band_indices.append(idx)
        band_names.append(name)
    
    all_bands_df = extract_raster_values_at_points(
        raster_path, filtered_samples, band_indices, band_names,
        band_mapping_config=raster_config.get("band_mapping_config")
    )
    
    if all_bands_df is None:
        raise ValueError("无法提取任何波段数据")
    
    # 4. 计算特征（30个：4波段+26指数）
    features_df = calculate_features_from_bands(all_bands_df)
    
    # 5. 提取盐分值
    salinity_col = data_config['column_names']['salinity']
    if salinity_col not in filtered_samples.columns:
        raise ValueError(f"样点数据中缺少盐分值列: {salinity_col}")
    salinity_values = filtered_samples[salinity_col].values
    
    # 确保长度匹配
    if len(salinity_values) != len(features_df):
        logger.warning(f"盐分值数量 ({len(salinity_values)}) 与特征数量 ({len(features_df)}) 不匹配")
        # 如果特征数量少于样点数量（可能因为某些样点提取失败），需要匹配
        if len(features_df) < len(salinity_values):
            # 只保留有效样点的盐分值
            salinity_values = salinity_values[:len(features_df)]
            logger.info("已截取盐分值以匹配特征数量")
        else:
            raise ValueError(f"无法匹配盐分值数量 ({len(salinity_values)}) 与特征数量 ({len(features_df)})")
    
    # 6. 数据清理
    valid_mask = ~(features_df.isna().any(axis=1) | np.isnan(salinity_values))
    features_df = features_df[valid_mask]
    salinity_values = salinity_values[valid_mask]
    
    logger.info(f"有效样本数: {len(features_df)}")
    
    # 7. 划分训练集和测试集
    X = features_df.values
    y = salinity_values
    feature_names = features_df.columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_RATIO, random_state=RANDOM_STATE
    )
    
    logger.info(f"训练集: {len(X_train)} 个样本")
    logger.info(f"测试集: {len(X_test)} 个样本")
    
    # 8. 特征选择
    logger.info("\n特征选择...")
    selected_features, selection_info = compare_and_select_features(
        X_train, y_train, feature_names
    )
    
    X_train_selected = X_train[:, selection_info["final_indices"]]
    X_test_selected = X_test[:, selection_info["final_indices"]]
    
    logger.info(f"特征选择完成: {len(selected_features)} 个特征")
    logger.info(f"选择策略: {selection_info['selection_strategy']}")
    
    # 9. 训练多个模型
    logger.info("\n训练多个模型...")
    model_results = train_multiple_models(
        X_train_selected, y_train, X_test_selected, y_test, use_grid_search=True
    )
    
    # 10. 评估模型
    logger.info("\n评估模型...")
    perf_list = []
    for model_name, result in model_results.items():
        y_pred = result["y_pred"]
        metrics = calculate_metrics(y_test, y_pred, model_name, len(selected_features))
        perf_list.append(metrics)
        
        logger.info(f"{model_name}: R²={metrics['R²']:.4f}, RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}")
        
        # 绘制散点图
        plot_scatter(
            y_test, y_pred, f"{exp_name} - {model_name}",
            output_dir / f"scatter_{model_name.lower()}.png"
        )
    
    perf_df = pd.DataFrame(perf_list)
    
    # 11. 保存结果
    logger.info("\n保存结果...")
    
    # 保存评估结果
    excel_path = output_dir / "experiment_results.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        perf_df.to_excel(writer, sheet_name="模型评估", index=False)
        
        # 特征选择结果
        feature_selection_df = pd.DataFrame({
            "特征": feature_names,
            "皮尔逊相关系数": selection_info["pearson_correlations"],
            "RFE排名": selection_info["rfe_ranking"],
            "是否选中": [f in selected_features for f in feature_names]
        })
        feature_selection_df.to_excel(writer, sheet_name="特征选择", index=False)
        
        # 元信息
        meta_df = pd.DataFrame({
            "键": ["实验名称", "生成时间", "样本数", "特征数", "选择策略"],
            "值": [
                exp_name,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                len(features_df),
                len(selected_features),
                selection_info["selection_strategy"]
            ]
        })
        meta_df.to_excel(writer, sheet_name="元信息", index=False)
    
    logger.info(f"结果已保存: {excel_path}")
    
    return {
        "experiment_name": exp_name,
        "performance": perf_df,
        "selected_features": selected_features,
        "selection_info": selection_info,
        "model_results": model_results,
        "output_dir": output_dir
    }


def main():
    """主函数：运行所有实验"""
    logger.info("=" * 60)
    logger.info("Stage 2 盐分反演对比实验")
    logger.info("=" * 60)
    
    # 加载配置
    data_config = load_config(CONFIG_PATH)
    
    # 读取样点数据
    samples_path = Path(data_config['data_paths']['samples_shapefile'])
    logger.info(f"读取样点数据: {samples_path}")
    samples_gdf = gpd.read_file(samples_path)
    logger.info(f"样点总数: {len(samples_gdf)}")
    
    # 更新实验配置中的栅格路径（仅对需要从config读取的实验）
    if EXPERIMENTS["exp1_uav_original"]["raster_path"] is None:
        EXPERIMENTS["exp1_uav_original"]["raster_path"] = Path(data_config['data_paths']['uav_raster'])
    if EXPERIMENTS["exp2_s2_original"]["raster_path"] is None:
        EXPERIMENTS["exp2_s2_original"]["raster_path"] = Path(data_config['data_paths']['s2_raster'])
    if EXPERIMENTS["exp3_l8_original"]["raster_path"] is None:
        EXPERIMENTS["exp3_l8_original"]["raster_path"] = Path(data_config['data_paths']['l8_raster'])
    
    # 运行所有实验
    all_results = {}
    for exp_key, exp_config in EXPERIMENTS.items():
        try:
            result = run_experiment(exp_config, data_config, samples_gdf)
            all_results[exp_key] = result
        except Exception as e:
            logger.error(f"实验 {exp_key} 失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 生成对比报告
    logger.info("\n生成对比报告...")
    comparison_df = pd.DataFrame()
    for exp_key, result in all_results.items():
        perf_df = result["performance"]
        perf_df["实验"] = result["experiment_name"]
        comparison_df = pd.concat([comparison_df, perf_df], ignore_index=True)
    
    comparison_path = OUTPUT_BASE_DIR / "experiments_comparison.xlsx"
    with pd.ExcelWriter(comparison_path, engine="openpyxl") as writer:
        comparison_df.to_excel(writer, sheet_name="所有实验对比", index=False)
    
    logger.info(f"对比报告已保存: {comparison_path}")
    logger.info("\n所有实验完成！")


if __name__ == "__main__":
    main()

