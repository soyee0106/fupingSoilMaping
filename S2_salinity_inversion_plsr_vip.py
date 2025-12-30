"""
基于 UAV 光谱特征的土壤盐分反演（皮尔逊相关性 + RFE特征选择 + 多模型）

流程包含：
1. 数据读取、特征选择（第 9 列起的 30 个 UAV 指数）
2. 皮尔逊相关性和RFE特征选择（比较两种方法）
3. 多个模型训练（SVR, RandomForest, PLSR, XGBoost, GradientBoosting）
4. 模型评估与可视化
5. 导出多 Sheet Excel + 预测结果
样点数据(138个)
    ↓ 过滤（排除富平FP样点）
    ↓ 提取30个光谱指数特征
    ↓
特征选择（双重策略）
    ├─ 皮尔逊：|r| >= 0.3
    └─ RFE：保留15个
    ↓ 智能融合（交集/并集）
    ↓
多模型训练
    ├─ SVR（网格搜索调参）
    ├─ RandomForest（200棵树）
    ├─ PLSR（自动选主成分）
    ├─ GradientBoosting
    └─ XGBoost（可选）
    ↓
模型评估
    ├─ R², RMSE, MAE等11项指标
    ├─ 散点图
    └─ 选最优模型
    ↓
空间预测（可选）
    ├─ 读取30波段影像
    ├─ 逐像元预测
    ├─ 过滤背景
    └─ 生成分布图
    ↓
结果输出
    ├─ Excel多表格报告
    ├─ 可视化图表
    └─ 预测栅格
"""

from __future__ import annotations
from datetime import datetime
from pathlib import Path

from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
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

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️ XGBoost未安装，将跳过XGBoost模型")

rcParams["font.family"] = ["SimHei"]
rcParams["axes.unicode_minus"] = False

# ================= 用户输入区域 =================
# input_csv = Path(
#   r"D:\富平星机光谱融合反演\middata\Sim_samples_extracted_values.csv_with_indices_v2.csv"
# )
input_csv = Path(
  r"D:\富平星机光谱融合反演\middata\S2_samples_extracted_values.csv_with_indices_.csv"
)
output_dir = Path(
    r"D:\富平星机光谱融合反演\结果v2\plsr\S2_plsr_vip_multimodel_全盐量_排除富平"
)
target_column = "全盐量"

feature_count = 30  # 选择最后30列作为特征
test_ratio = 0.2
random_state = 42

# 样本过滤参数（根据列中包含的字符串排除样本）
filter_column = "抽样编"  # 用于过滤的列名，如果为 None 则不进行过滤
exclude_strings = ["FP"]  # 要排除的字符串列表，例如：["A1", "B2"] 表示排除列值中包含 "A1" 或 "B2" 的样本

# 特征选择参数
pearson_threshold = 0.3  # 皮尔逊相关系数阈值（绝对值）
rfe_n_features = 15  # RFE选择的特征数量
rfe_estimator = "RandomForest"  # RFE使用的估计器类型: "RandomForest", "SVR", "PLSR"
cv_folds = 5  # 交叉验证折数

# SVR GridSearch参数
svr_param_grid = {
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
    "epsilon": [0.01, 0.1, 0.5],
}

excel_name = "salinity_results_plsr_vip.xlsx"

# 空间分布图生成参数
generate_spatial_map = False  # 是否生成空间分布图
raster_image_path = Path(
    r"D:\富平星机光谱融合反演\数据\S2_features_30bands.tif"
)  # 用于预测的栅格影像路径
spatial_map_model = "GradientBoosting"  # 用于空间预测的模型名称（从训练好的模型中选择，如 "SVR", "RandomForest", "PLSR" 等）
output_salinity_raster = output_dir / "S2salinity_predictionGradientBoosting.tif"  # 输出的盐分预测栅格路径

# 背景像元去除参数
remove_background_pixels = True  # 是否去除背景像元
background_threshold = None  # 背景阈值（如果所有波段值都小于此值，则视为背景），None表示不启用
check_nodata_values = True  # 是否检查原始栅格的nodata值
# =================================================


def ensure_paths() -> None:
    if not input_csv.exists():
        raise FileNotFoundError(f"未找到输入 CSV：{input_csv}")
    output_dir.mkdir(parents=True, exist_ok=True)


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """选择最后30列作为特征"""
    if len(df.columns) < feature_count:
        raise ValueError(
            f"列数不足：需要至少 {feature_count} 列，当前 {len(df.columns)} 列。"
        )
    features = df.iloc[:, -feature_count:]
    if features.shape[1] != feature_count:
        raise ValueError(f"特征数量不足 {feature_count} 列，请检查输入数据。")
    return features


def select_features_by_pearson(
    X: np.ndarray, y: np.ndarray, feature_names: list, threshold: float = 0.3
) -> tuple[list, np.ndarray]:
    """
    基于皮尔逊相关系数选择特征
    
    参数:
        X: 特征矩阵
        y: 目标变量
        feature_names: 特征名称列表
        threshold: 相关系数阈值（绝对值）
    
    返回:
        selected_features: 选中的特征名称列表
        selected_indices: 选中的特征索引
        correlation_scores: 所有特征的相关系数
    """
    print(f"\n📊 基于皮尔逊相关系数选择特征（阈值={threshold}）...")
    
    correlations = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        corr, _ = pearsonr(X[:, i], y)
        correlations[i] = corr
    
    # 选择绝对值大于阈值的特征
    abs_correlations = np.abs(correlations)
    selected_mask = abs_correlations >= threshold
    selected_indices = np.where(selected_mask)[0]
    
    # 如果选中的特征太少，则选择前top_n个
    if len(selected_indices) < 5:
        print(f"  ⚠️ 阈值筛选后特征数过少（{len(selected_indices)}个），改为选择前15个")
        top_n = min(15, X.shape[1])
        selected_indices = np.argsort(abs_correlations)[::-1][:top_n]
        selection_method = f"前{top_n}个"
    else:
        selection_method = f"|r|>={threshold}"
    
    selected_features = [feature_names[i] for i in selected_indices]
    
    print(f"  ✅ 皮尔逊相关性选择：{len(selected_features)} 个特征（{selection_method}）")
    print(f"  相关系数范围：[{correlations[selected_indices].min():.3f}, {correlations[selected_indices].max():.3f}]")
    
    return selected_features, selected_indices, correlations


def select_features_by_rfe(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    n_features: int = 15,
    estimator_type: str = "RandomForest",
) -> tuple[list, np.ndarray]:
    """
    基于递归特征消除（RFE）选择特征
    
    参数:
        X: 特征矩阵
        y: 目标变量
        feature_names: 特征名称列表
        n_features: 要选择的特征数量
        estimator_type: 估计器类型 ("RandomForest", "SVR", "PLSR")
    
    返回:
        selected_features: 选中的特征名称列表
        selected_indices: 选中的特征索引
        rfe_ranking: RFE特征排名（1表示最重要）
    """
    print(f"\n📊 基于RFE选择特征（目标特征数={n_features}，估计器={estimator_type}）...")
    
    # 选择估计器
    if estimator_type == "RandomForest":
        estimator = RandomForestRegressor(
            n_estimators=100, random_state=random_state, n_jobs=-1
        )
    elif estimator_type == "SVR":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        estimator = SVR(kernel="rbf", C=10, gamma="scale", epsilon=0.1)
        X = X_scaled
    elif estimator_type == "PLSR":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        estimator = PLSRegression(n_components=min(10, X.shape[1], X.shape[0]))
        X = X_scaled
    else:
        raise ValueError(f"不支持的估计器类型: {estimator_type}")
    
    # 限制特征数量不超过样本数和特征数
    n_features = min(n_features, X.shape[0], X.shape[1])
    
    # 执行RFE
    rfe = RFE(estimator=estimator, n_features_to_select=n_features, step=1)
    rfe.fit(X, y)
    
    # 获取选中的特征
    selected_mask = rfe.support_
    selected_indices = np.where(selected_mask)[0]
    selected_features = [feature_names[i] for i in selected_indices]
    
    # 获取特征排名（1表示最重要）
    rfe_ranking = rfe.ranking_
    
    print(f"  ✅ RFE选择：{len(selected_features)} 个特征")
    print(f"  特征排名范围：[{rfe_ranking[selected_indices].min()}, {rfe_ranking[selected_indices].max()}]")
    
    return selected_features, selected_indices, rfe_ranking


def compare_and_select_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    pearson_threshold: float = 0.3,
    rfe_n_features: int = 15,
    rfe_estimator: str = "RandomForest",
) -> tuple[list, dict]:
    """
    比较皮尔逊相关性和RFE两种方法，选择最终特征
    
    参数:
        X: 特征矩阵
        y: 目标变量
        feature_names: 特征名称列表
        pearson_threshold: 皮尔逊相关系数阈值
        rfe_n_features: RFE选择的特征数量
        rfe_estimator: RFE使用的估计器类型
    
    返回:
        final_selected_features: 最终选中的特征名称列表
        selection_info: 选择信息字典
    """
    print("\n" + "=" * 60)
    print("特征选择：比较皮尔逊相关性和RFE")
    print("=" * 60)
    
    # 方法1：皮尔逊相关性
    pearson_features, pearson_indices, pearson_correlations = select_features_by_pearson(
        X, y, feature_names, threshold=pearson_threshold
    )
    
    # 方法2：RFE
    rfe_features, rfe_indices, rfe_ranking = select_features_by_rfe(
        X, y, feature_names, n_features=rfe_n_features, estimator_type=rfe_estimator
    )
    
    # 比较两种方法
    print("\n📊 方法比较：")
    print(f"  皮尔逊相关性选择：{len(pearson_features)} 个特征")
    print(f"  RFE选择：{len(rfe_features)} 个特征")
    
    # 计算交集和并集
    pearson_set = set(pearson_features)
    rfe_set = set(rfe_features)
    intersection = pearson_set & rfe_set
    union = pearson_set | rfe_set
    
    print(f"  交集：{len(intersection)} 个特征")
    print(f"  并集：{len(union)} 个特征")
    
    # 选择策略：优先使用交集，如果交集太少则使用并集
    if len(intersection) >= 5:
        final_selected_features = list(intersection)
        selection_strategy = "交集"
        print(f"\n✅ 使用交集策略：{len(final_selected_features)} 个特征")
    else:
        final_selected_features = list(union)
        selection_strategy = "并集"
        print(f"\n✅ 交集特征数过少，使用并集策略：{len(final_selected_features)} 个特征")
    
    # 获取最终特征的索引
    final_indices = [feature_names.index(f) for f in final_selected_features]
    
    # 构建选择信息
    selection_info = {
        "pearson_features": pearson_features,
        "pearson_indices": pearson_indices,
        "pearson_correlations": pearson_correlations,
        "rfe_features": rfe_features,
        "rfe_indices": rfe_indices,
        "rfe_ranking": rfe_ranking,
        "intersection": list(intersection),
        "union": list(union),
        "final_features": final_selected_features,
        "final_indices": final_indices,
        "selection_strategy": selection_strategy,
    }
    
    return final_selected_features, selection_info


def train_multiple_models(X_train, y_train, X_test, y_test, use_grid_search=True):
    """训练多个模型"""
    results = {}
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 1. SVR
    print("\n📊 训练SVR模型...")
    if use_grid_search:
        svr = SVR(kernel="rbf")
        grid_search = GridSearchCV(
            svr,
            svr_param_grid,
            cv=cv_folds,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            verbose=0,
        )
        grid_search.fit(X_train_scaled, y_train)
        svr_model = grid_search.best_estimator_
        print(f"  最优参数：{grid_search.best_params_}")
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
    print("\n📊 训练RandomForest模型...")
    rf_model = RandomForestRegressor(
        n_estimators=200, random_state=random_state, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    results["RandomForest"] = {
        "model": rf_model,
        "scaler": None,
        "y_pred": y_pred_rf,
        "needs_scaling": False,
    }
    
    # 3. PLSR Baseline
    print("\n📊 训练PLSR Baseline模型...")
    # PLSR的主成分数不能超过特征数或样本数的最小值
    max_components = min(X_train_scaled.shape[0], X_train_scaled.shape[1])
    n_comp_range = range(1, min(21, max_components + 1))
    
    best_score = -np.inf
    optimal_n_comp = 1
    for n_comp in n_comp_range:
        plsr_temp = PLSRegression(n_components=n_comp)
        cv_scores = cross_val_score(
            plsr_temp, X_train_scaled, y_train, cv=cv_folds, scoring="neg_mean_squared_error", n_jobs=-1
        )
        mean_score = cv_scores.mean()
        if mean_score > best_score:
            best_score = mean_score
            optimal_n_comp = n_comp
    
    print(f"  最优主成分数：{optimal_n_comp}")
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
    print("\n📊 训练GradientBoosting模型...")
    gb_model = GradientBoostingRegressor(
        n_estimators=100, random_state=random_state, max_depth=5
    )
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
        print("\n📊 训练XGBoost模型...")
        xgb_model = XGBRegressor(
            n_estimators=100, random_state=random_state, n_jobs=-1
        )
        xgb_model.fit(X_train, y_train)
        y_pred_xgb = xgb_model.predict(X_test)
        results["XGBoost"] = {
            "model": xgb_model,
            "scaler": None,
            "y_pred": y_pred_xgb,
            "needs_scaling": False,
        }
    
    return results


def metrics_row(y_true, y_pred, name, n_features):
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
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolor="k")
    plt.plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        "r--",
        lw=2,
        label="1:1 线",
    )
    plt.xlabel("真实全盐 (g/kg)")
    plt.ylabel("预测全盐 (g/kg)")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  ✅ 散点图已保存：{save_path}")


def plot_heatmap(df: pd.DataFrame, features, target):
    data = df[[target] + features]
    corr = data.corr(method="pearson")

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr,
        annot=True,
        cmap="RdYlGn",
        center=0,
        fmt=".2f",
        cbar=True,
        square=True,
        linewidths=0.5,
        annot_kws={"size": 8},
    )
    plt.title(f"{target} 与光谱指数的 Pearson 相关系数", fontsize=14)
    plt.tight_layout()
    out_path = output_dir / "pearson_corr_heatmap.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✅ Pearson 热力图已保存：{out_path}")


def apply_model_to_raster(
    model,
    scaler,
    selected_features: list,
    feature_columns: list,
    raster_path: Path,
    output_path: Path,
    needs_scaling: bool = True,
    remove_background: bool = True,
    background_threshold: float | None = None,
    check_nodata: bool = True,
) -> np.ndarray:
    """
    将训练好的模型应用到栅格影像，生成盐分预测栅格
    
    参数:
        model: 训练好的模型
        scaler: StandardScaler（如果使用标准化）
        selected_features: 选中的特征名称列表
        feature_columns: 原始特征列名列表（用于匹配栅格波段）
        raster_path: 输入的栅格影像路径
        output_path: 输出的预测栅格路径
        needs_scaling: 模型是否需要标准化
    
    返回:
        预测结果数组
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
        print(f"   Nodata: {nodata}")
        
        # 读取所有波段数据
        all_bands = src.read()  # (bands, height, width)
        print("✅ 影像读取完成")
    
    # 检查波段数是否匹配特征数
    if count < len(selected_features):
        raise ValueError(
            f"栅格波段数 ({count}) 少于需要的特征数 ({len(selected_features)})"
        )
    
    print("\n📊 提取特征波段...")
    print(f"   需要特征: {len(selected_features)} 个")
    
    # 假设栅格波段的顺序对应特征列的顺序（从最后feature_count列开始）
    # 例如，如果feature_count=30，则使用最后30个波段
    if count >= feature_count:
        band_start_idx = count - feature_count
        band_indices = list(range(band_start_idx, count))
    else:
        band_indices = list(range(count))
    
    print(f"   使用波段索引: {band_indices}")
    
    # 构建特征矩阵：选择对应的波段
    feature_data = []
    feature_band_mapping = {}
    
    for feat_name in selected_features:
        # 找到该特征在原始特征列中的索引
        if feat_name in feature_columns:
            feat_idx = feature_columns.index(feat_name)
            # 找到对应的波段索引（假设从后往前对应）
            band_idx = band_start_idx + (len(feature_columns) - 1 - feat_idx)
            if band_idx < count:
                feature_data.append(all_bands[band_idx, :, :])
                feature_band_mapping[feat_name] = band_idx
                print(f"      {feat_name} → 波段 {band_idx + 1}")
            else:
                raise ValueError(f"特征 {feat_name} 无法匹配到有效的波段")
        else:
            print(f"⚠️ 警告：特征 {feat_name} 不在特征列中，跳过")
    
    if len(feature_data) == 0:
        raise ValueError("没有找到任何匹配的特征波段")
    
    print(f"✅ 成功提取 {len(feature_data)} 个特征波段")
    
    # 构建特征矩阵：(height, width, n_features)
    feature_stack = np.stack(feature_data, axis=2)  # (height, width, n_features)
    feature_flat = feature_stack.reshape(-1, len(feature_data))  # (n_pixels, n_features)
    
    print(f"   特征矩阵形状: {feature_flat.shape}")
    
    # 识别背景像元
    print("\n🔍 识别背景像元...")
    
    # 1. 检查无效值（NaN, Inf）
    valid_mask = np.all(np.isfinite(feature_flat), axis=1)
    invalid_count = (~valid_mask).sum()
    if invalid_count > 0:
        print(f"   无效值（NaN/Inf）: {invalid_count} 个像素")
    
    # 2. 检查原始nodata值（如果启用）
    background_mask = np.zeros(len(feature_flat), dtype=bool)
    if check_nodata and nodata is not None:
        # 检查特征波段中是否有nodata值
        nodata_mask = np.any(feature_flat == nodata, axis=1)
        background_mask = background_mask | nodata_mask
        nodata_count = nodata_mask.sum()
        if nodata_count > 0:
            print(f"   包含nodata值: {nodata_count} 个像素")
    
    # 3. 检查背景阈值（如果启用）
    if background_threshold is not None:
        # 如果所有特征波段的值都小于阈值，视为背景
        low_value_mask = np.all(feature_flat < background_threshold, axis=1)
        background_mask = background_mask | low_value_mask
        low_value_count = low_value_mask.sum()
        if low_value_count > 0:
            print(f"   低于阈值 ({background_threshold}): {low_value_count} 个像素")
    
    # 4. 检查所有特征值都为零或接近零的像素（可能是背景）
    zero_mask = np.all(np.abs(feature_flat) < 1e-6, axis=1)
    background_mask = background_mask | zero_mask
    zero_count = zero_mask.sum()
    if zero_count > 0:
        print(f"   所有值接近零: {zero_count} 个像素")
    
    # 合并所有背景像元判断
    if remove_background:
        # 有效像素 = 非无效值 且 非背景像元
        final_valid_mask = valid_mask & (~background_mask)
        background_total = (~final_valid_mask).sum()
        print(f"\n   背景像元总数: {background_total} 个像素")
    else:
        final_valid_mask = valid_mask
        background_total = 0
        print("\n   背景像元去除: 已禁用")
    
    feature_flat_valid = feature_flat[final_valid_mask]
    
    print(f"   最终有效像素数: {final_valid_mask.sum()}/{len(final_valid_mask)} "
          f"({100*final_valid_mask.sum()/len(final_valid_mask):.2f}%)")
    
    # 标准化（如果需要）
    if needs_scaling and scaler is not None:
        print("   标准化特征...")
        feature_scaled = scaler.transform(feature_flat_valid)
    else:
        feature_scaled = feature_flat_valid
    
    # 预测
    print("   应用模型预测...")
    predicted_valid = model.predict(feature_scaled)
    
    # 重塑回影像形状
    predicted_band = np.full((height, width), np.nan, dtype=np.float32)
    predicted_band_flat = predicted_band.ravel()
    predicted_band_flat[final_valid_mask] = predicted_valid
    predicted_band = predicted_band_flat.reshape(height, width)
    
    # 计算统计信息（直接使用 predicted_valid，它已经是有效像素的预测值）
    print(f"   预测值范围: [{predicted_valid.min():.4f}, {predicted_valid.max():.4f}]")
    print(f"   预测值均值: {predicted_valid.mean():.4f}")
    print(f"   预测值标准差: {predicted_valid.std():.4f}")
    
    # 保存预测栅格
    print("\n💾 保存预测栅格...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    nodata_value = -9999.0
    
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=np.float32,
        crs=crs,
        transform=transform,
        nodata=nodata_value,
        compress='lzw',
    ) as dst:
        # 将NaN替换为nodata值
        band_data_clean = np.where(np.isfinite(predicted_band), predicted_band, nodata_value)
        dst.write(band_data_clean, 1)
    
    print(f"✅ 预测栅格已保存：{output_path}")
    
    return predicted_band


def plot_salinity_spatial_distribution(
    raster_path: Path,
    output_path: Path,
    title: str = "土壤盐分空间分布",
    cmap_name: str = "YlOrRd",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """
    绘制盐分空间分布图（使用地理坐标，保持正确的宽高比）
    
    参数:
        raster_path: 盐分预测栅格路径
        output_path: 输出图像路径
        title: 图标题
        cmap_name: 颜色映射名称
        vmin: 最小值（用于颜色映射）
        vmax: 最大值（用于颜色映射）
    """
    print("\n🎨 生成空间分布图...")
    
    if not raster_path.exists():
        raise FileNotFoundError(f"未找到栅格文件：{raster_path}")
    
    # 读取栅格
    with rasterio.open(raster_path) as src:
        salinity_data = src.read(1)  # 读取第一个波段
        crs = src.crs
        nodata = src.nodata
        bounds = src.bounds
        
        # 处理nodata值
        if nodata is not None:
            salinity_data = np.where(salinity_data == nodata, np.nan, salinity_data)
        
        # 获取有效数据范围
        valid_data = salinity_data[np.isfinite(salinity_data)]
        if len(valid_data) == 0:
            raise ValueError("栅格中没有有效数据")
        
        if vmin is None:
            vmin = np.nanpercentile(salinity_data, 2)  # 使用2%分位数作为最小值
        if vmax is None:
            vmax = np.nanpercentile(salinity_data, 98)  # 使用98%分位数作为最大值
        
        # 计算地理范围（用于设置extent）
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
        
        # 计算宽高比（基于地理坐标）
        width_geo = bounds.right - bounds.left
        height_geo = bounds.top - bounds.bottom
        aspect_ratio = width_geo / height_geo if height_geo > 0 else 1.0
    
    # 创建图形（根据宽高比调整图形大小）
    fig_width = 12
    fig_height = fig_width / aspect_ratio
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # 创建掩码，排除NaN值
    masked_data = np.ma.masked_invalid(salinity_data)
    
    # 绘制栅格（使用地理坐标extent）
    im = ax.imshow(
        masked_data,
        cmap=cmap_name,
        vmin=vmin,
        vmax=vmax,
        interpolation='bilinear',
        extent=extent,  # 使用地理坐标范围
        aspect='equal',  # 保持等比例，这样地理坐标才能正确显示
    )
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('全盐量 (g/kg)', fontsize=12, rotation=270, labelpad=20)
    
    # 设置标题和标签（使用地理坐标）
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # 根据CRS设置坐标轴标签
    if crs is not None:
        if crs.is_geographic:
            ax.set_xlabel('经度 (°)', fontsize=12)
            ax.set_ylabel('纬度 (°)', fontsize=12)
        else:
            ax.set_xlabel(f'X 坐标 ({crs.linear_units})', fontsize=12)
            ax.set_ylabel(f'Y 坐标 ({crs.linear_units})', fontsize=12)
    else:
        ax.set_xlabel('X 坐标', fontsize=12)
        ax.set_ylabel('Y 坐标', fontsize=12)
    
    # 添加统计信息文本框
    stats_text = (
        f"最小值: {valid_data.min():.2f} g/kg\n"
        f"最大值: {valid_data.max():.2f} g/kg\n"
        f"平均值: {valid_data.mean():.2f} g/kg\n"
        f"标准差: {valid_data.std():.2f} g/kg\n"
        f"有效像素: {len(valid_data):,}"
    )
    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
    )
    
    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    
    plt.tight_layout()
    
    # 保存图像
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 空间分布图已保存：{output_path}")
    print(f"   图像尺寸: {fig_width:.1f} × {fig_height:.1f} 英寸 (宽高比: {aspect_ratio:.3f})")
    print(f"   地理范围: X=[{bounds.left:.6f}, {bounds.right:.6f}], Y=[{bounds.bottom:.6f}, {bounds.top:.6f}]")


def build_descriptive_stats_table(
    df: pd.DataFrame,
    train_idx: pd.Index,
    test_idx: pd.Index,
    salt_col: str = "全盐 (g/kg)",
    ec_col: str | None = "电导率（ds/m)",
) -> pd.DataFrame:
    """构建描述性统计分析表"""
    
    # 全部样本
    salt_all = df[salt_col]
    has_ec = ec_col is not None and ec_col in df.columns
    ec_all = df[ec_col] if has_ec else None
    
    # 建模样本（训练集）
    salt_train = df.loc[train_idx, salt_col]
    ec_train = df.loc[train_idx, ec_col] if has_ec else None
    
    # 验证样本（测试集）
    salt_test = df.loc[test_idx, salt_col]
    ec_test = df.loc[test_idx, ec_col] if has_ec else None
    
    # 构建统计表
    stats_list = []
    
    # 全盐统计
    stats_list.append({
        "统计指标": "数量",
        "全部样本": len(salt_all.dropna()),
        "建模样本": len(salt_train.dropna()),
        "验证样本": len(salt_test.dropna()),
    })
    stats_list.append({
        "统计指标": f"平均值({salt_col})",
        "全部样本": salt_all.mean(),
        "建模样本": salt_train.mean(),
        "验证样本": salt_test.mean(),
    })
    stats_list.append({
        "统计指标": f"最大值({salt_col})",
        "全部样本": salt_all.max(),
        "建模样本": salt_train.max(),
        "验证样本": salt_test.max(),
    })
    stats_list.append({
        "统计指标": f"最小值({salt_col})",
        "全部样本": salt_all.min(),
        "建模样本": salt_train.min(),
        "验证样本": salt_test.min(),
    })
    stats_list.append({
        "统计指标": f"标准差({salt_col})",
        "全部样本": salt_all.std(),
        "建模样本": salt_train.std(),
        "验证样本": salt_test.std(),
    })
    cv_all = (salt_all.std() / salt_all.mean()) * 100 if salt_all.mean() != 0 else np.nan
    cv_train = (salt_train.std() / salt_train.mean()) * 100 if salt_train.mean() != 0 else np.nan
    cv_test = (salt_test.std() / salt_test.mean()) * 100 if salt_test.mean() != 0 else np.nan
    stats_list.append({
        "统计指标": f"变异系数({salt_col})",
        "全部样本": cv_all,
        "建模样本": cv_train,
        "验证样本": cv_test,
    })
    
    # 电导率统计（如果存在）
    if has_ec:
        stats_list.append({
            "统计指标": "数量",
            "全部样本": len(ec_all.dropna()),
            "建模样本": len(ec_train.dropna()),
            "验证样本": len(ec_test.dropna()),
        })
        stats_list.append({
            "统计指标": f"平均值({ec_col})",
            "全部样本": ec_all.mean(),
            "建模样本": ec_train.mean(),
            "验证样本": ec_test.mean(),
        })
        stats_list.append({
            "统计指标": f"最大值({ec_col})",
            "全部样本": ec_all.max(),
            "建模样本": ec_train.max(),
            "验证样本": ec_test.max(),
        })
        stats_list.append({
            "统计指标": f"最小值({ec_col})",
            "全部样本": ec_all.min(),
            "建模样本": ec_train.min(),
            "验证样本": ec_test.min(),
        })
        stats_list.append({
            "统计指标": f"标准差({ec_col})",
            "全部样本": ec_all.std(),
            "建模样本": ec_train.std(),
            "验证样本": ec_test.std(),
        })
        cv_ec_all = (ec_all.std() / ec_all.mean()) * 100 if ec_all.mean() != 0 else np.nan
        cv_ec_train = (ec_train.std() / ec_train.mean()) * 100 if ec_train.mean() != 0 else np.nan
        cv_ec_test = (ec_test.std() / ec_test.mean()) * 100 if ec_test.mean() != 0 else np.nan
        stats_list.append({
            "统计指标": f"变异系数({ec_col})",
            "全部样本": cv_ec_all,
            "建模样本": cv_ec_train,
            "验证样本": cv_ec_test,
        })
    
    return pd.DataFrame(stats_list)


def main() -> None:
    ensure_paths()

    print("=" * 60)
    print("UAV光谱盐分反演：皮尔逊相关性 + RFE特征选择 + 多模型")
    print("=" * 60)
    print("📌 配置信息：")
    print(f"  - 目标列：{target_column}")
    print(f"  - 皮尔逊相关系数阈值：{pearson_threshold}")
    print(f"  - RFE特征数：{rfe_n_features}")
    print(f"  - RFE估计器：{rfe_estimator}")

    df = pd.read_csv(input_csv)
    if target_column not in df.columns:
        raise KeyError(f"目标列 {target_column} 不存在。")

    # 根据列中包含的字符串过滤样本
    if filter_column is not None and exclude_strings:
        if filter_column not in df.columns:
            print(f"⚠️ 警告：过滤列 '{filter_column}' 不存在，跳过过滤。")
        else:
            original_count = len(df)
            print("\n📊 样本过滤（根据列中包含的字符串）...")
            print(f"   过滤列：{filter_column}")
            print(f"   原始样本数：{original_count}")
            print(f"   排除字符串：{exclude_strings}")
            
            # 创建过滤掩码：排除列值中包含指定字符串的样本
            exclude_mask = pd.Series([False] * len(df), index=df.index)
            
            # 统计每个字符串的数量
            string_counts = {}
            
            for exclude_str in exclude_strings:
                # 检查列值中是否包含该字符串
                str_mask = df[filter_column].astype(str).str.contains(exclude_str, na=False, regex=False)
                exclude_mask = exclude_mask | str_mask
                string_counts[exclude_str] = str_mask.sum()
            
            excluded_count = exclude_mask.sum()
            
            # 过滤数据
            filtered_df = df[~exclude_mask].copy()
            df = filtered_df.reset_index(drop=True)
            
            print(f"   排除样本数：{excluded_count}")
            print(f"   保留样本数：{len(df)}")
            print("   排除的字符串统计：")
            for exclude_str in exclude_strings:
                count = string_counts.get(exclude_str, 0)
                print(f"      '{exclude_str}': {count} 个样本")
            
            print("✅ 过滤完成")

    X = select_features(df)
    y = df[target_column]

    # 移除缺失值
    valid_mask = ~y.isna()
    X = X[valid_mask]
    y = y[valid_mask]
    df = df[valid_mask].reset_index(drop=True)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    print(f"\n✅ 数据加载完成：{len(X)} 个样本，{X.shape[1]} 个特征")

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=random_state
    )

    # 获取训练/测试集索引用于描述性统计
    train_idx = X_train.index
    test_idx = X_test.index

    print("\n📊 数据划分：")
    print(f"  - 训练集：{X_train.shape[0]} 个样本")
    print(f"  - 测试集：{X_test.shape[0]} 个样本")

    # ========== 特征选择（皮尔逊相关性 + RFE） ==========
    print("\n" + "=" * 60)
    print("步骤1：特征选择（皮尔逊相关性 + RFE）")
    print("=" * 60)

    # 转换为numpy数组用于特征选择
    X_train_array = X_train.values
    feature_names = X_train.columns.tolist()

    # 比较并选择特征
    selected_features, selection_info = compare_and_select_features(
        X_train_array,
        y_train.values,
        feature_names,
        pearson_threshold=pearson_threshold,
        rfe_n_features=rfe_n_features,
        rfe_estimator=rfe_estimator,
    )

    # 提取选中的特征
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    print(f"✅ 特征选择完成：{X_train_selected.shape[1]} 个特征")
    print(f"   选择策略：{selection_info['selection_strategy']}")

    # ========== 多模型训练 ==========
    print("\n" + "=" * 60)
    print("步骤2：多模型训练")
    print("=" * 60)

    # 在筛选后的特征上训练多个模型
    model_results = train_multiple_models(
        X_train_selected, y_train, X_test_selected, y_test, use_grid_search=True
    )

    # ========== 模型评估 ==========
    print("\n" + "=" * 60)
    print("步骤3：模型评估")
    print("=" * 60)

    perf_list = []
    for model_name, result in model_results.items():
        y_pred = result["y_pred"]
        metrics = metrics_row(y_test, y_pred, model_name, X_train_selected.shape[1])
        perf_list.append(metrics)
        
        print(f"\n📊 {model_name} 测试集结果：")
        print(f"  RMSE: {metrics['RMSE']:.4f}")
        print(f"  MAE: {metrics['MAE']:.4f}")
        print(f"  R²: {metrics['R²']:.4f}")
        print(f"  Pearson_r: {metrics['Pearson_r']:.4f}")

    perf_df = pd.DataFrame(perf_list)

    # ========== 可视化 ==========
    print("\n" + "=" * 60)
    print("步骤4：可视化")
    print("=" * 60)

    # 绘制散点图
    for model_name, result in model_results.items():
        plot_scatter(
            y_test,
            result["y_pred"],
            f"{model_name} (特征筛选)",
            output_dir / f"scatter_{model_name.lower()}.png",
        )

    # 绘制特征选择结果对比图
    print("\n📊 绘制特征选择结果对比图...")
    
    # 创建特征选择结果DataFrame
    feature_selection_df = pd.DataFrame({
        "特征": feature_names,
        "皮尔逊相关系数": selection_info["pearson_correlations"],
        "RFE排名": selection_info["rfe_ranking"],
    })
    feature_selection_df["是否选中"] = feature_selection_df["特征"].isin(selected_features)
    feature_selection_df = feature_selection_df.sort_values("皮尔逊相关系数", key=abs, ascending=False)
    
    # 绘制皮尔逊相关系数条形图
    plt.figure(figsize=(12, 8))
    top_features = feature_selection_df.head(20)
    colors = ['red' if sel else 'blue' for sel in top_features["是否选中"]]
    plt.barh(range(len(top_features)), top_features["皮尔逊相关系数"].values, color=colors)
    plt.yticks(range(len(top_features)), top_features["特征"].values)
    plt.xlabel("皮尔逊相关系数")
    plt.title("Top 20 特征皮尔逊相关系数（红色=选中，蓝色=未选中）")
    plt.axvline(x=pearson_threshold, color='green', linestyle='--', label=f'阈值={pearson_threshold}')
    plt.axvline(x=-pearson_threshold, color='green', linestyle='--')
    plt.legend()
    plt.gca().invert_yaxis()
    plt.tight_layout()
    pearson_plot_path = output_dir / "pearson_correlation_values.png"
    plt.savefig(pearson_plot_path, dpi=300)
    plt.close()
    print(f"✅ 皮尔逊相关系数条形图已保存：{pearson_plot_path}")
    
    # 绘制RFE排名条形图
    plt.figure(figsize=(12, 8))
    rfe_df = feature_selection_df.sort_values("RFE排名")
    top_rfe = rfe_df.head(20)
    colors_rfe = ['red' if sel else 'blue' for sel in top_rfe["是否选中"]]
    plt.barh(range(len(top_rfe)), top_rfe["RFE排名"].values, color=colors_rfe)
    plt.yticks(range(len(top_rfe)), top_rfe["特征"].values)
    plt.xlabel("RFE排名（1=最重要）")
    plt.title("Top 20 特征RFE排名（红色=选中，蓝色=未选中）")
    plt.legend()
    plt.gca().invert_yaxis()
    plt.tight_layout()
    rfe_plot_path = output_dir / "rfe_ranking.png"
    plt.savefig(rfe_plot_path, dpi=300)
    plt.close()
    print(f"✅ RFE排名条形图已保存：{rfe_plot_path}")

    # 绘制Pearson热力图
    plot_heatmap(df, X.columns.tolist(), target_column)

    # ========== 保存预测结果 ==========
    print("\n" + "=" * 60)
    print("步骤5：保存结果")
    print("=" * 60)

    # 生成预测结果DataFrame
    df_pred = df.copy()
    for model_name, result in model_results.items():
        model = result["model"]
        scaler = result["scaler"]
        needs_scaling = result["needs_scaling"]
        
        if needs_scaling and scaler is not None:
            X_all_scaled = scaler.transform(X[selected_features])
            df_pred[f"Pred_{model_name}"] = model.predict(X_all_scaled)
        else:
            df_pred[f"Pred_{model_name}"] = model.predict(X[selected_features])

    # 生成描述性统计分析表
    ec_col = "电导率（ds/m)"
    if ec_col not in df.columns:
        possible_ec_cols = [
            col for col in df.columns
            if "电导率" in str(col) or "EC" in str(col).upper()
        ]
        if possible_ec_cols:
            ec_col = possible_ec_cols[0]
            print(f"⚠️ 使用 '{ec_col}' 作为电导率列")
        else:
            print("⚠️ 未找到电导率列，仅统计全盐数据")
            ec_col = None

    desc_stats_df = build_descriptive_stats_table(
        df, train_idx, test_idx, target_column, ec_col
    )

    # 元信息
    meta_df = pd.DataFrame(
        {
            "键": [
                "生成时间",
                "样本数",
                "原始特征数",
                "筛选后特征数",
                "皮尔逊相关性特征数",
                "RFE特征数",
                "交集特征数",
                "选择策略",
                "皮尔逊阈值",
                "RFE特征数",
                "RFE估计器",
            ],
            "值": [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                len(df),
                X.shape[1],
                X_train_selected.shape[1],
                len(selection_info["pearson_features"]),
                len(selection_info["rfe_features"]),
                len(selection_info["intersection"]),
                selection_info["selection_strategy"],
                pearson_threshold,
                rfe_n_features,
                rfe_estimator,
            ],
        }
    )

    # 保存Excel
    excel_path = output_dir / excel_name
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        feature_selection_df.to_excel(writer, sheet_name="特征选择结果", index=False)
        
        # 皮尔逊相关性结果
        pearson_df = pd.DataFrame({
            "特征": feature_names,
            "皮尔逊相关系数": selection_info["pearson_correlations"],
            "是否选中": [f in selection_info["pearson_features"] for f in feature_names]
        }).sort_values("皮尔逊相关系数", key=abs, ascending=False)
        pearson_df.to_excel(writer, sheet_name="皮尔逊相关性", index=False)
        
        # RFE结果
        rfe_df = pd.DataFrame({
            "特征": feature_names,
            "RFE排名": selection_info["rfe_ranking"],
            "是否选中": [f in selection_info["rfe_features"] for f in feature_names]
        }).sort_values("RFE排名")
        rfe_df.to_excel(writer, sheet_name="RFE排名", index=False)
        
        # 方法比较
        comparison_df = pd.DataFrame({
            "特征": feature_names,
            "皮尔逊选中": [f in selection_info["pearson_features"] for f in feature_names],
            "RFE选中": [f in selection_info["rfe_features"] for f in feature_names],
            "交集": [f in selection_info["intersection"] for f in feature_names],
            "最终选中": [f in selected_features for f in feature_names],
        })
        comparison_df.to_excel(writer, sheet_name="方法比较", index=False)
        
        perf_df.to_excel(writer, sheet_name="模型评估", index=False)
        meta_df.to_excel(writer, sheet_name="元信息", index=False)
        df_pred.to_excel(writer, sheet_name="预测结果", index=False)
        desc_stats_df.to_excel(writer, sheet_name="描述性统计分析", index=False)

    print(f"✅ 结果 Excel 已输出：{excel_path}")

    # ========== 生成空间分布图 ==========
    if generate_spatial_map:
        print("\n" + "=" * 60)
        print("步骤6：生成空间分布图")
        print("=" * 60)
        
        if spatial_map_model not in model_results:
            print(f"⚠️ 警告：模型 '{spatial_map_model}' 不存在，可用模型: {list(model_results.keys())}")
            print("   跳过空间分布图生成")
        else:
            result = model_results[spatial_map_model]
            model = result["model"]
            scaler = result["scaler"]
            needs_scaling = result["needs_scaling"]
            
            print(f"   使用模型: {spatial_map_model}")
            print(f"   输入影像: {raster_image_path}")
            
            try:
                # 应用模型到栅格
                apply_model_to_raster(
                    model=model,
                    scaler=scaler,
                    selected_features=selected_features,
                    feature_columns=X.columns.tolist(),
                    raster_path=raster_image_path,
                    output_path=output_salinity_raster,
                    needs_scaling=needs_scaling,
                    remove_background=remove_background_pixels,
                    background_threshold=background_threshold,
                    check_nodata=check_nodata_values,
                )
                
                # 生成空间分布图
                spatial_map_path = output_dir / f"salinity_spatial_distribution_{spatial_map_model}.png"
                plot_salinity_spatial_distribution(
                    raster_path=output_salinity_raster,
                    output_path=spatial_map_path,
                    title=f"土壤盐分空间分布图 ({spatial_map_model} 模型预测)",
                )
                
                print("\n✅ 空间分布图生成完成！")
                print(f"   预测栅格: {output_salinity_raster}")
                print(f"   分布图: {spatial_map_path}")
                
            except Exception as e:
                print(f"❌ 生成空间分布图时出错: {e}")
                import traceback
                traceback.print_exc()

    print("\n✅ 处理完成！")


if __name__ == "__main__":
    main()

