"""
数据配对模块

核心功能：从配准后的多源影像中提取样点位置的光谱值，
生成包含所有特征和标签的DataFrame，用于模型训练。
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from pathlib import Path
from typing import Dict, List, Optional
import logging
import sys
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from scipy import stats

# 处理相对导入问题：既可以作为模块导入，也可以直接运行
try:
    from .band_matching import create_band_mapping
    from .spectral_indices import calculate_all_indices
except ImportError:
    # 如果相对导入失败，尝试绝对导入（用于直接运行）
    # 添加父目录到路径
    current_dir = Path(__file__).parent
    parent_dir = current_dir.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    from data_preprocessing.band_matching import create_band_mapping
    from data_preprocessing.spectral_indices import calculate_all_indices

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _plot_distributions(
    data_before: np.ndarray,
    data_after: np.ndarray,
    sensor_name: str,
    band_names: List[str],
    scaler,
    output_dir: Path
):
    """
    绘制标准化前后的分布图。
    
    参数:
        data_before: 标准化前的数据，形状为 (n_samples, n_bands)
        data_after: 标准化后的数据，形状为 (n_samples, n_bands)
        sensor_name: 传感器名称（S2, L8, UAV）
        band_names: 波段名称列表
        scaler: 标准化器对象
        output_dir: 输出目录
    """
    n_bands = data_before.shape[1]
    n_cols = 2  # 两列：标准化前和标准化后
    n_rows = n_bands
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建大图
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_bands))
    if n_bands == 1:
        axes = axes.reshape(1, -1)
    
    for band_idx in range(n_bands):
        band_name = band_names[band_idx] if band_idx < len(band_names) else f'Band{band_idx+1}'
        
        # 提取该波段的数据
        data_before_band = data_before[:, band_idx]
        data_after_band = data_after[:, band_idx]
        
        # 移除NaN值
        data_before_band = data_before_band[~np.isnan(data_before_band)]
        data_after_band = data_after_band[~np.isnan(data_after_band)]
        
        # 标准化前的分布
        ax_before = axes[band_idx, 0]
        ax_before.hist(data_before_band, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        
        # 拟合正态分布曲线
        if len(data_before_band) > 0:
            mu_before = np.mean(data_before_band)
            sigma_before = np.std(data_before_band)
            x_before = np.linspace(data_before_band.min(), data_before_band.max(), 100)
            y_before = stats.norm.pdf(x_before, mu_before, sigma_before)
            ax_before.plot(x_before, y_before, 'r-', linewidth=2, label=f'Normal(μ={mu_before:.2f}, σ={sigma_before:.2f})')
            ax_before.legend()
        
        ax_before.set_xlabel('Value', fontsize=10)
        ax_before.set_ylabel('Density', fontsize=10)
        ax_before.set_title(f'{sensor_name} {band_name} - Before Normalization\n'
                          f'Mean={np.mean(data_before_band):.4f}, Std={np.std(data_before_band):.4f}',
                          fontsize=11)
        ax_before.grid(True, alpha=0.3)
        
        # 标准化后的分布
        ax_after = axes[band_idx, 1]
        ax_after.hist(data_after_band, bins=50, density=True, alpha=0.7, color='lightgreen', edgecolor='black')
        
        # 拟合标准正态分布曲线（均值0，标准差1）
        if len(data_after_band) > 0:
            mu_after = np.mean(data_after_band)
            sigma_after = np.std(data_after_band)
            x_after = np.linspace(data_after_band.min(), data_after_band.max(), 100)
            y_after = stats.norm.pdf(x_after, mu_after, sigma_after)
            ax_after.plot(x_after, y_after, 'r-', linewidth=2, label=f'Normal(μ={mu_after:.2f}, σ={sigma_after:.2f})')
            
            # 绘制标准正态分布参考线（如果接近标准正态）
            if abs(mu_after) < 0.1 and abs(sigma_after - 1.0) < 0.1:
                x_std = np.linspace(-4, 4, 100)
                y_std = stats.norm.pdf(x_std, 0, 1)
                ax_after.plot(x_std, y_std, 'b--', linewidth=1.5, alpha=0.7, label='Standard Normal(0,1)')
            
            ax_after.legend()
        
        ax_after.set_xlabel('Normalized Value', fontsize=10)
        ax_after.set_ylabel('Density', fontsize=10)
        ax_after.set_title(f'{sensor_name} {band_name} - After Normalization (Z-score)\n'
                          f'Mean={np.mean(data_after_band):.4f}, Std={np.std(data_after_band):.4f}',
                          fontsize=11)
        ax_after.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    output_path = output_dir / f'{sensor_name}_distribution_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"    已保存 {sensor_name} 分布图: {output_path}")


class DataPairer:
    """
    数据配对类，用于从多源影像中提取样点光谱值并生成训练数据。
    
    功能：
    1. 读取样点坐标（Shapefile格式）
    2. 从配准后的卫星影像（S2, L8）中提取光谱值
    3. 从无人机影像中提取光谱值
    4. 计算光谱指数
    5. 合并所有特征和标签，生成DataFrame
    """
    
    def __init__(
        self,
        band_mapping: Optional[Dict] = None,
        sensor_labels: Optional[Dict[str, int]] = None
    ):
        """
        初始化DataPairer。
        
        参数:
            band_mapping: 波段映射字典，如果为None则使用默认映射
            sensor_labels: 传感器标签字典，如 {'S2': 0, 'L8': 1}
        """
        if band_mapping is None:
            self.band_mapping = create_band_mapping()
        else:
            self.band_mapping = band_mapping
        
        if sensor_labels is None:
            self.sensor_labels = {'S2': 0, 'L8': 1}
        else:
            self.sensor_labels = sensor_labels
        
        logger.info("DataPairer initialized")
        logger.info(f"Band mapping: {self.band_mapping['uav_bands']}")
        logger.info(f"Sensor labels: {self.sensor_labels}")
    
    def extract_raster_values(
        self,
        raster_path: Path,
        gdf: gpd.GeoDataFrame,
        band_names: Optional[List[str]] = None,
        band_indices: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        从栅格影像中提取样点位置的像元值。
        
        参数:
            raster_path: 栅格影像路径
            gdf: 包含样点几何信息的GeoDataFrame
            band_names: 波段列名列表，如果为None则自动生成
            band_indices: 要提取的波段索引列表（从1开始），如果为None则提取所有波段
        
        返回:
            DataFrame，包含提取的像元值
        """
        logger.info(f"Extracting values from raster: {raster_path}")
        
        if not raster_path.exists():
            raise FileNotFoundError(f"Raster file not found: {raster_path}")
        
        with rasterio.open(raster_path) as src:
            num_bands = src.count
            crs = src.crs
            
            logger.info(f"  Total bands: {num_bands}, CRS: {crs}, Size: {src.height}x{src.width}")
            
            # 尝试读取波段描述信息
            band_descriptions = {}
            if hasattr(src, 'descriptions') and src.descriptions:
                for i, desc in enumerate(src.descriptions, 1):
                    if desc:
                        band_descriptions[i] = desc
                        logger.info(f"  Band {i} description: {desc}")
            
            # 确定要提取的波段索引
            if band_indices is None:
                # 提取所有波段
                band_indices_to_extract = list(range(1, num_bands + 1))
            else:
                # 只提取指定的波段
                band_indices_to_extract = band_indices
                # 验证索引有效性
                for idx in band_indices_to_extract:
                    if idx < 1 or idx > num_bands:
                        raise ValueError(f"Band index {idx} is out of range (1-{num_bands})")
            
            logger.info(f"  Extracting bands: {band_indices_to_extract}")
            
            # 检查CRS一致性
            if gdf.crs != crs:
                logger.warning(f"  CRS mismatch: sample CRS ({gdf.crs}) != raster CRS ({crs}), reprojecting...")
                gdf_reprojected = gdf.to_crs(crs)
            else:
                gdf_reprojected = gdf.copy()
            
            # 提取样点位置的像元值
            extracted_values = []
            
            for idx, row in gdf_reprojected.iterrows():
                geom = row.geometry
                sample_values = []
                
                for band_idx in band_indices_to_extract:
                    values = list(src.sample([(geom.x, geom.y)], indexes=[band_idx]))
                    if values:
                        sample_values.append(values[0][0])
                    else:
                        sample_values.append(np.nan)
                
                extracted_values.append(sample_values)
            
            # 创建DataFrame
            if band_names is None:
                band_columns = [f"band_{i}" for i in band_indices_to_extract]
            else:
                if len(band_names) != len(band_indices_to_extract):
                    raise ValueError(f"Number of band names ({len(band_names)}) doesn't match number of bands ({len(band_indices_to_extract)})")
                band_columns = band_names
            
            values_df = pd.DataFrame(extracted_values, columns=band_columns)
            
            logger.info(f"  Successfully extracted values for {len(values_df)} samples")
        
        return values_df
    
    def find_bands_by_mapping(
        self,
        raster_path: Path,
        band_mapping: Dict,
        target_bands: List[str] = ['G', 'R', 'REG', 'NIR']
    ) -> Dict[str, int]:
        """
        根据波段映射配置，从栅格影像中查找目标波段的索引。
        
        参数:
            raster_path: 栅格影像路径
            band_mapping: 波段映射配置字典（从YAML配置中读取）
            target_bands: 目标波段列表，如['G', 'R', 'REG', 'NIR']
        
        返回:
            字典，键为目标波段名，值为波段索引（从1开始）
        """
        result = {}
        
        with rasterio.open(raster_path) as src:
            num_bands = src.count
            
            # 尝试读取波段描述
            band_descriptions = {}
            if hasattr(src, 'descriptions') and src.descriptions:
                for i, desc in enumerate(src.descriptions, 1):
                    if desc:
                        band_descriptions[i] = desc.lower()  # 转为小写便于匹配
            
            # 为每个目标波段查找对应的索引
            for target_band in target_bands:
                if target_band not in band_mapping:
                    logger.warning(f"  Target band '{target_band}' not found in mapping config")
                    continue
                
                mapping = band_mapping[target_band]
                found_index = None
                
                # 方法1：使用配置的索引
                if mapping.get('indices') and len(mapping['indices']) > 0:
                    for idx in mapping['indices']:
                        if 1 <= idx <= num_bands:
                            found_index = idx
                            logger.info(f"  Found {target_band} at index {idx} (from config)")
                            break
                
                # 方法2：根据波段描述匹配
                if found_index is None and mapping.get('names'):
                    for band_idx, desc in band_descriptions.items():
                        for name in mapping['names']:
                            if name.lower() in desc:
                                found_index = band_idx
                                logger.info(f"  Found {target_band} at index {band_idx} (matched description: {desc})")
                                break
                        if found_index:
                            break
                
                if found_index:
                    result[target_band] = found_index
                else:
                    logger.warning(f"  Could not find band '{target_band}', will use 0 or skip")
                    result[target_band] = None
        
        return result
    
    def extract_satellite_features(
        self,
        s2_raster_path: Optional[Path] = None,
        l8_raster_path: Optional[Path] = None,
        samples_gdf: gpd.GeoDataFrame = None,
        band_mapping_config: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        从卫星影像中提取特征（波段值和光谱指数）。
        
        参数:
            s2_raster_path: Sentinel-2影像路径
            l8_raster_path: Landsat-8影像路径
            samples_gdf: 样点GeoDataFrame
        
        返回:
            DataFrame，包含卫星特征和传感器标签
        """
        all_features = []
        sensor_types = []
        
        # 处理Sentinel-2数据
        if s2_raster_path is not None and s2_raster_path.exists():
            logger.info("Processing Sentinel-2 data...")
            
            # 根据配置查找波段索引
            s2_mapping = None
            if band_mapping_config and 's2' in band_mapping_config:
                s2_mapping = band_mapping_config['s2'].get('band_mapping', {})
            
            if s2_mapping:
                # 使用配置的波段映射自动识别
                band_indices_map = self.find_bands_by_mapping(
                    s2_raster_path,
                    s2_mapping,
                    target_bands=['G', 'R', 'REG', 'NIR']
                )
                
                # 按目标波段顺序提取找到的波段
                target_band_order = ['G', 'R', 'REG', 'NIR']
                band_indices = []
                band_names = []
                for target_band in target_band_order:
                    if target_band in band_indices_map and band_indices_map[target_band] is not None:
                        band_indices.append(band_indices_map[target_band])
                        band_names.append(f"S2_{target_band}")
                
                if band_indices:
                    s2_bands_raw = self.extract_raster_values(
                        s2_raster_path,
                        samples_gdf,
                        band_names=band_names,
                        band_indices=band_indices
                    )
                    
                    # 构建统一格式的波段DataFrame（按目标顺序）
                    s2_bands = pd.DataFrame()
                    for i, target_band in enumerate(target_band_order):
                        if target_band in band_indices_map and band_indices_map[target_band] is not None:
                            # 从提取的数据中获取对应列
                            col_name = f"S2_{target_band}"
                            if col_name in s2_bands_raw.columns:
                                s2_bands[target_band] = s2_bands_raw[col_name].values
                            else:
                                # 如果列名不匹配，使用索引位置
                                s2_bands[target_band] = s2_bands_raw.iloc[:, i].values
                        else:
                            # REG波段缺失，填充0
                            s2_bands[target_band] = np.zeros(len(s2_bands_raw))
                else:
                    # 如果没有找到任何波段，使用默认值
                    logger.warning("  No bands found, using zeros")
                    s2_bands = pd.DataFrame({
                        'G': np.zeros(len(samples_gdf)),
                        'R': np.zeros(len(samples_gdf)),
                        'REG': np.zeros(len(samples_gdf)),
                        'NIR': np.zeros(len(samples_gdf))
                    })
            else:
                # 回退到默认方式：假设波段顺序固定
                logger.warning("  No band mapping config found, using default indices")
                s2_bands_raw = self.extract_raster_values(
                    s2_raster_path,
                    samples_gdf,
                    band_names=['S2_B2', 'S2_B3', 'S2_B4', 'S2_B8'],
                    band_indices=[2, 3, 4, 8]  # S2的B2, B3, B4, B8
                )
                
                # 统一格式为G, R, REG, NIR
                s2_bands = pd.DataFrame({
                    'G': s2_bands_raw['S2_B3'].values,  # S2的B3（绿光）对应G
                    'R': s2_bands_raw['S2_B4'].values,  # S2的B4（红光）对应R
                    'REG': np.zeros(len(s2_bands_raw)),  # S2没有REG，填充0
                    'NIR': s2_bands_raw['S2_B8'].values,  # S2的B8（近红外）对应NIR
                })
            
            # 生成掩码向量：1表示有效，0表示缺失
            # S2: G=1, R=1, REG=0, NIR=1
            s2_band_mask = np.array([[1, 1, 0, 1]] * len(s2_bands))
            
            # 计算光谱指数（使用G, R, REG, NIR格式）
            bands_dict = {
                'G': s2_bands['G'].values,
                'R': s2_bands['R'].values,
                'REG': s2_bands['REG'].values,
                'NIR': s2_bands['NIR'].values,
            }
            indices = calculate_all_indices(bands_dict)
            
            # 合并波段和指数
            s2_features = s2_bands.copy()
            for idx_name, idx_values in indices.items():
                s2_features[f'{idx_name}'] = idx_values
            
            # 添加传感器标签和掩码
            s2_features['sensor'] = self.sensor_labels['S2']
            # 将掩码向量作为字符串存储（每行一个掩码向量）
            s2_features['band_mask'] = [','.join(map(str, mask)) for mask in s2_band_mask]
            
            all_features.append(s2_features)
            sensor_types.extend(['S2'] * len(s2_features))
        
        # 处理Landsat-8数据
        if l8_raster_path is not None and l8_raster_path.exists():
            logger.info("Processing Landsat-8 data...")
            
            # 根据配置查找波段索引
            l8_mapping = None
            if band_mapping_config and 'l8' in band_mapping_config:
                l8_mapping = band_mapping_config['l8'].get('band_mapping', {})
            
            if l8_mapping:
                # 使用配置的波段映射自动识别
                band_indices_map = self.find_bands_by_mapping(
                    l8_raster_path,
                    l8_mapping,
                    target_bands=['G', 'R', 'REG', 'NIR']
                )
                
                # 按目标波段顺序提取找到的波段
                target_band_order = ['G', 'R', 'REG', 'NIR']
                band_indices = []
                band_names = []
                for target_band in target_band_order:
                    if target_band in band_indices_map and band_indices_map[target_band] is not None:
                        band_indices.append(band_indices_map[target_band])
                        band_names.append(f"L8_{target_band}")
                
                if band_indices:
                    l8_bands_raw = self.extract_raster_values(
                        l8_raster_path,
                        samples_gdf,
                        band_names=band_names,
                        band_indices=band_indices
                    )
                    
                    # 构建统一格式的波段DataFrame（按目标顺序）
                    l8_bands = pd.DataFrame()
                    for i, target_band in enumerate(target_band_order):
                        if target_band in band_indices_map and band_indices_map[target_band] is not None:
                            # 从提取的数据中获取对应列
                            col_name = f"L8_{target_band}"
                            if col_name in l8_bands_raw.columns:
                                l8_bands[target_band] = l8_bands_raw[col_name].values
                            else:
                                # 如果列名不匹配，使用索引位置
                                l8_bands[target_band] = l8_bands_raw.iloc[:, i].values
                        else:
                            # REG波段缺失，填充0
                            l8_bands[target_band] = np.zeros(len(l8_bands_raw))
                else:
                    # 如果没有找到任何波段，使用默认值
                    logger.warning("  No bands found, using zeros")
                    l8_bands = pd.DataFrame({
                        'G': np.zeros(len(samples_gdf)),
                        'R': np.zeros(len(samples_gdf)),
                        'REG': np.zeros(len(samples_gdf)),
                        'NIR': np.zeros(len(samples_gdf))
                    })
            else:
                # 回退到默认方式：假设波段顺序固定
                logger.warning("  No band mapping config found, using default indices")
                l8_bands_raw = self.extract_raster_values(
                    l8_raster_path,
                    samples_gdf,
                    band_names=['L8_B2', 'L8_B3', 'L8_B4', 'L8_B5'],
                    band_indices=[2, 3, 4, 5]  # L8的B2, B3, B4, B5
                )
                
                # 统一格式为G, R, REG, NIR
                l8_bands = pd.DataFrame({
                    'G': l8_bands_raw['L8_B3'].values,  # L8的B3（绿光）对应G
                    'R': l8_bands_raw['L8_B4'].values,  # L8的B4（红光）对应R
                    'REG': np.zeros(len(l8_bands_raw)),  # L8没有REG，填充0
                    'NIR': l8_bands_raw['L8_B5'].values,  # L8的B5（近红外）对应NIR
                })
            
            # 生成掩码向量：1表示有效，0表示缺失
            # L8: G=1, R=1, REG=0, NIR=1
            l8_band_mask = np.array([[1, 1, 0, 1]] * len(l8_bands))
            
            # 计算光谱指数（使用G, R, REG, NIR格式）
            bands_dict = {
                'G': l8_bands['G'].values,
                'R': l8_bands['R'].values,
                'REG': l8_bands['REG'].values,
                'NIR': l8_bands['NIR'].values,
            }
            indices = calculate_all_indices(bands_dict)
            
            # 合并波段和指数
            l8_features = l8_bands.copy()
            for idx_name, idx_values in indices.items():
                l8_features[f'{idx_name}'] = idx_values
            
            # 添加传感器标签和掩码
            l8_features['sensor'] = self.sensor_labels['L8']
            # 将掩码向量作为字符串存储（每行一个掩码向量）
            l8_features['band_mask'] = [','.join(map(str, mask)) for mask in l8_band_mask]
            
            all_features.append(l8_features)
            sensor_types.extend(['L8'] * len(l8_features))
        
        if not all_features:
            raise ValueError("No satellite data provided!")
        
        # 合并所有特征
        result_df = pd.concat(all_features, axis=0, ignore_index=True)
        logger.info(f"Total satellite features extracted: {len(result_df)} samples")
        
        return result_df
    
    def extract_uav_features(
        self,
        uav_raster_path: Path,
        samples_gdf: gpd.GeoDataFrame
    ) -> pd.DataFrame:
        """
        从无人机影像中提取特征（4个波段和光谱指数）。
        
        参数:
            uav_raster_path: 无人机影像路径
            samples_gdf: 样点GeoDataFrame
        
        返回:
            DataFrame，包含无人机波段和光谱指数
        """
        logger.info("Processing UAV data...")
        
        uav_bands = self.extract_raster_values(
            uav_raster_path,
            samples_gdf,
            band_names=['UAV_G', 'UAV_R', 'UAV_REG', 'UAV_NIR']
        )
        
        # 计算光谱指数
        bands_dict = {
            'G': uav_bands['UAV_G'].values,
            'R': uav_bands['UAV_R'].values,
            'REG': uav_bands['UAV_REG'].values,
            'NIR': uav_bands['UAV_NIR'].values,
        }
        indices = calculate_all_indices(bands_dict)
        
        # 合并波段和指数
        uav_features = uav_bands.copy()
        for idx_name, idx_values in indices.items():
            uav_features[f'UAV_{idx_name}'] = idx_values
        
        logger.info(f"UAV features extracted: {len(uav_features)} samples")
        
        return uav_features
    
    def create_training_data(
        self,
        samples_shapefile: Path,
        salinity_column: str,
        s2_raster_path: Optional[Path] = None,
        l8_raster_path: Optional[Path] = None,
        uav_raster_path: Optional[Path] = None,
        sample_id_column: Optional[str] = None,
        band_mapping_config: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        创建完整的训练数据集。
        
        参数:
            samples_shapefile: 样点Shapefile路径
            salinity_column: 盐分值列名
            s2_raster_path: Sentinel-2影像路径
            l8_raster_path: Landsat-8影像路径
            uav_raster_path: 无人机影像路径
            sample_id_column: 样点ID列名，用于匹配数据
        
        返回:
            DataFrame，包含所有特征和标签
        """
        logger.info("=" * 60)
        logger.info("Creating training dataset")
        logger.info("=" * 60)
        
        # 1. 读取样点数据
        logger.info(f"Reading samples from: {samples_shapefile}")
        samples_gdf = gpd.read_file(samples_shapefile)
        logger.info(f"  Loaded {len(samples_gdf)} samples")
        
        # 2. 提取卫星特征
        satellite_features = self.extract_satellite_features(
            s2_raster_path=s2_raster_path,
            l8_raster_path=l8_raster_path,
            samples_gdf=samples_gdf,
            band_mapping_config=band_mapping_config
        )
        
        # 3. 提取无人机特征（如果提供）
        uav_features = None
        if uav_raster_path is not None and uav_raster_path.exists():
            uav_features = self.extract_uav_features(uav_raster_path, samples_gdf)
        
        # 4. 合并数据
        # 首先合并样点属性（包括盐分值）
        sample_attrs = samples_gdf.drop(columns=['geometry']).copy()
        if 'geometry' in samples_gdf.columns:
            sample_attrs['longitude'] = samples_gdf.geometry.x
            sample_attrs['latitude'] = samples_gdf.geometry.y
        
        # 根据样点数量匹配数据
        n_samples = len(samples_gdf)
        
        # 如果卫星特征数量是样点数量的倍数（多个传感器），需要处理
        if len(satellite_features) > n_samples:
            # 多个传感器的情况，需要复制样点属性
            n_sensors = len(satellite_features) // n_samples
            sample_attrs_repeated = pd.concat([sample_attrs] * n_sensors, ignore_index=True)
            result_df = pd.concat([sample_attrs_repeated, satellite_features], axis=1)
        else:
            result_df = pd.concat([sample_attrs, satellite_features], axis=1)
        
        # 添加无人机特征（如果提供）
        if uav_features is not None:
            if len(uav_features) == len(result_df):
                result_df = pd.concat([result_df, uav_features], axis=1)
            elif len(uav_features) == n_samples and len(result_df) > n_samples:
                # 无人机特征需要重复以匹配多传感器数据
                n_sensors = len(result_df) // n_samples
                uav_features_repeated = pd.concat([uav_features] * n_sensors, ignore_index=True)
                result_df = pd.concat([result_df, uav_features_repeated], axis=1)
            else:
                logger.warning("UAV features length mismatch, skipping UAV features")
        
        # 5. 检查盐分值列
        if salinity_column not in result_df.columns:
            raise ValueError(f"Salinity column '{salinity_column}' not found in data")
        
        logger.info(f"Final dataset shape: {result_df.shape}")
        logger.info(f"Columns: {list(result_df.columns)}")
        
        return result_df
    
    def create_dense_training_data(
        self,
        s2_raster_path: Path,
        l8_raster_path: Optional[Path] = None,
        uav_raster_path: Path = None,
        target_resolution: Optional[float] = None,
        s2_band_indices: Optional[List[int]] = None,
        l8_band_indices: Optional[List[int]] = None,
        uav_band_indices: Optional[List[int]] = None,
        s2_nodata: Optional[float] = None,
        l8_nodata: Optional[float] = None,
        uav_nodata: Optional[float] = None,
        band_mapping_config: Optional[Dict] = None,
        output_l8_tif: Optional[Path] = None,
        output_uav_tif: Optional[Path] = None,
        output_csv: Optional[Path] = None,
        sensor_labels: Optional[Dict[str, int]] = None,
        normalize: bool = True,
        output_scalers_dir: Optional[Path] = None,
        output_normalized_l8_tif: Optional[Path] = None,
        output_normalized_uav_tif: Optional[Path] = None,
        output_normalized_s2_tif: Optional[Path] = None,
        output_normalized_csv: Optional[Path] = None,
        normalization_method: str = "standard",
        output_distribution_plots_dir: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        创建密集训练数据集：对齐影像并提取所有有效像元对。
        
        处理流程：
        1. UAV影像聚合到S2尺度
        2. Landsat-8重采样到S2尺度
        3. 对齐到S2像元网格
        4. 生成配对数据：(卫星观测光谱, 传感器ID, 无人机本征光谱真值)
        5. 保存对齐后的影像和配对数据表
        
        参数:
            s2_raster_path: Sentinel-2影像路径（作为参考网格）
            l8_raster_path: Landsat-8影像路径（可选）
            uav_raster_path: 无人机影像路径
            target_resolution: 目标分辨率（如果为None，使用S2分辨率）
            s2_band_indices: S2要提取的波段索引（从1开始）
            l8_band_indices: L8要提取的波段索引（从1开始）
            uav_band_indices: UAV要提取的波段索引（从1开始）
            s2_nodata: S2的nodata值
            l8_nodata: L8的nodata值
            uav_nodata: UAV的nodata值
            band_mapping_config: 波段映射配置
            output_l8_tif: 输出对齐后的L8影像路径（可选）
            output_uav_tif: 输出对齐后的UAV影像路径（可选）
            output_csv: 输出配对数据CSV路径（可选）
            sensor_labels: 传感器标签字典，如 {'S2': 0, 'L8': 1}
        
        返回:
            DataFrame，包含配对数据：(卫星光谱, 传感器ID, UAV光谱)
        """
        logger.info("=" * 60)
        logger.info("Creating Dense Training Dataset (Pixel-to-Pixel Mapping)")
        logger.info("=" * 60)
        
        if sensor_labels is None:
            sensor_labels = self.sensor_labels
        
        from .image_alignment_export import (
            aggregate_uav_to_s2_resolution,
            align_to_s2_grid,
            save_aligned_raster
        )
        import rasterio
        
        # 读取S2影像信息（作为参考网格）
        with rasterio.open(s2_raster_path) as s2_src:
            s2_crs = s2_src.crs
            s2_transform = s2_src.transform
            s2_height = s2_src.height
            s2_width = s2_src.width
            
            if target_resolution is None:
                target_resolution = abs(s2_transform[0])
            
            logger.info(f"S2影像信息: {s2_height} × {s2_width}, CRS: {s2_crs}")
            logger.info(f"目标分辨率: {target_resolution}")
        
        # 检查必需的数据
        if not uav_raster_path or not uav_raster_path.exists():
            raise ValueError("UAV影像路径不存在！")
        if not l8_raster_path or not l8_raster_path.exists():
            raise ValueError("L8影像路径不存在！需要同时有S2、L8和UAV才能生成配对数据。")
        
        all_pairs = []
        
        # 1. 处理UAV：聚合对齐到S2
        logger.info("\n处理UAV影像...")
        with rasterio.open(uav_raster_path) as uav_src:
            uav_crs = uav_src.crs
            uav_transform = uav_src.transform
            uav_pixel_size_x = abs(uav_transform[0])
            uav_pixel_size_y = abs(uav_transform[4])
            
            if uav_band_indices is None:
                uav_band_indices = list(range(1, uav_src.count + 1))
            
            uav_bands_raw = []
            for idx in uav_band_indices:
                uav_bands_raw.append(uav_src.read(idx))
        
        # 聚合UAV到S2分辨率
        uav_bands_aggregated, uav_agg_transform = aggregate_uav_to_s2_resolution(
            uav_band_arrays=uav_bands_raw,
            uav_transform=uav_transform,
            uav_pixel_size_x=uav_pixel_size_x,
            uav_pixel_size_y=uav_pixel_size_y,
            target_resolution=target_resolution,
            nodata_value=uav_nodata
        )
        
        # 对齐UAV到S2网格
        uav_bands_aligned, uav_aligned_transform, uav_output_shape = align_to_s2_grid(
            source_bands=uav_bands_aggregated,
            source_transform=uav_agg_transform,
            source_crs=uav_crs,
            s2_raster_path=s2_raster_path,
            src_nodata=uav_nodata
        )
        
        # 保存UAV对齐影像
        if output_uav_tif:
            logger.info(f"保存UAV对齐影像: {output_uav_tif}")
            save_aligned_raster(
                aligned_bands=uav_bands_aligned,
                output_path=output_uav_tif,
                transform=uav_aligned_transform,
                crs=s2_crs,
                dtype="float32",
                nodata=None
            )
        
        uav_height, uav_width = uav_bands_aligned[0].shape
        logger.info(f"UAV对齐后尺寸: {uav_height} × {uav_width}")
        
        # 2. 处理L8：重采样对齐到S2
        logger.info("\n处理Landsat-8影像...")
        with rasterio.open(l8_raster_path) as l8_src:
            l8_crs = l8_src.crs
            l8_transform = l8_src.transform
            
            if l8_band_indices is None:
                l8_band_indices = list(range(1, l8_src.count + 1))
            
            l8_bands_raw = []
            for idx in l8_band_indices:
                l8_bands_raw.append(l8_src.read(idx))
        
        # 对齐L8到S2网格（直接重采样，不需要聚合）
        l8_bands_aligned, l8_aligned_transform, l8_output_shape = align_to_s2_grid(
            source_bands=l8_bands_raw,
            source_transform=l8_transform,
            source_crs=l8_crs,
            s2_raster_path=s2_raster_path,
            src_nodata=l8_nodata
        )
        
        # 保存L8对齐影像
        if output_l8_tif:
            logger.info(f"保存L8对齐影像: {output_l8_tif}")
            save_aligned_raster(
                aligned_bands=l8_bands_aligned,
                output_path=output_l8_tif,
                transform=l8_aligned_transform,
                crs=s2_crs,
                dtype="float32",
                nodata=None
            )
        
        l8_height, l8_width = l8_bands_aligned[0].shape
        logger.info(f"L8对齐后尺寸: {l8_height} × {l8_width}")
        
        # 3. 处理S2：读取对应区域
        logger.info("\n处理Sentinel-2影像...")
        with rasterio.open(s2_raster_path) as s2_src:
            if s2_band_indices is None:
                s2_band_indices = list(range(1, s2_src.count + 1))
            
            # 计算对齐区域在S2中的位置（使用UAV对齐后的transform）
            uav_ul_x = uav_aligned_transform[2]
            uav_ul_y = uav_aligned_transform[5]
            
            row_start, col_start = rasterio.transform.rowcol(s2_transform, uav_ul_x, uav_ul_y)
            row_end = min(s2_height, row_start + uav_height)
            col_end = min(s2_width, col_start + uav_width)
            
            # 读取S2对应区域
            s2_window = rasterio.windows.Window(col_start, row_start, col_end - col_start, row_end - row_start)
            s2_bands_aligned = []
            for idx in s2_band_indices:
                s2_band = s2_src.read(idx, window=s2_window)
                # 确保尺寸匹配
                if s2_band.shape != (uav_height, uav_width):
                    s2_band = s2_band[:uav_height, :uav_width]
                s2_bands_aligned.append(s2_band)
        
        s2_height_aligned, s2_width_aligned = s2_bands_aligned[0].shape
        logger.info(f"S2对齐后尺寸: {s2_height_aligned} × {s2_width_aligned}")
        
        # 4. 找到三者的交集区域（同时有S2、L8、UAV有效值的区域）
        logger.info("\n计算三者交集区域...")
        
        # 关键：需要找到L8对齐影像中对应UAV对齐区域的位置
        # UAV对齐后的transform和L8对齐后的transform应该都对齐到S2网格
        # 计算UAV对齐区域在L8对齐影像中的位置
        uav_ul_x = uav_aligned_transform[2]
        uav_ul_y = uav_aligned_transform[5]
        uav_lr_x = uav_ul_x + uav_width * uav_aligned_transform[0]
        uav_lr_y = uav_ul_y + uav_height * uav_aligned_transform[4]
        
        # 计算L8对齐影像的bounds
        l8_ul_x = l8_aligned_transform[2]
        l8_ul_y = l8_aligned_transform[5]
        l8_lr_x = l8_ul_x + l8_width * l8_aligned_transform[0]
        l8_lr_y = l8_ul_y + l8_height * l8_aligned_transform[4]
        
        logger.info(f"UAV对齐区域: UL=({uav_ul_x:.6f}, {uav_ul_y:.6f}), LR=({uav_lr_x:.6f}, {uav_lr_y:.6f})")
        logger.info(f"L8对齐区域: UL=({l8_ul_x:.6f}, {l8_ul_y:.6f}), LR=({l8_lr_x:.6f}, {l8_lr_y:.6f})")
        
        # 计算重叠区域
        overlap_ul_x = max(uav_ul_x, l8_ul_x)
        overlap_ul_y = min(uav_ul_y, l8_ul_y)  # Y坐标是反的
        overlap_lr_x = min(uav_lr_x, l8_lr_x)
        overlap_lr_y = max(uav_lr_y, l8_lr_y)  # Y坐标是反的
        
        logger.info(f"重叠区域: UL=({overlap_ul_x:.6f}, {overlap_ul_y:.6f}), LR=({overlap_lr_x:.6f}, {overlap_lr_y:.6f})")
        
        # 计算在UAV对齐影像中的位置（相对于UAV对齐区域）
        uav_row_start, uav_col_start = rasterio.transform.rowcol(
            uav_aligned_transform, overlap_ul_x, overlap_ul_y
        )
        uav_row_end, uav_col_end = rasterio.transform.rowcol(
            uav_aligned_transform, overlap_lr_x, overlap_lr_y
        )
        
        # 计算在L8对齐影像中的位置
        l8_row_start, l8_col_start = rasterio.transform.rowcol(
            l8_aligned_transform, overlap_ul_x, overlap_ul_y
        )
        l8_row_end, l8_col_end = rasterio.transform.rowcol(
            l8_aligned_transform, overlap_lr_x, overlap_lr_y
        )
        
        # 确保索引在有效范围内
        uav_row_start = max(0, uav_row_start)
        uav_col_start = max(0, uav_col_start)
        uav_row_end = min(uav_height, uav_row_end + 1)
        uav_col_end = min(uav_width, uav_col_end + 1)
        
        l8_row_start = max(0, l8_row_start)
        l8_col_start = max(0, l8_col_start)
        l8_row_end = min(l8_height, l8_row_end + 1)
        l8_col_end = min(l8_width, l8_col_end + 1)
        
        # 确定共同尺寸
        common_height = min(uav_row_end - uav_row_start, l8_row_end - l8_row_start, s2_height_aligned)
        common_width = min(uav_col_end - uav_col_start, l8_col_end - l8_col_start, s2_width_aligned)
        
        logger.info(f"UAV裁剪区域: 行[{uav_row_start}, {uav_row_end}), 列[{uav_col_start}, {uav_col_end})")
        logger.info(f"L8裁剪区域: 行[{l8_row_start}, {l8_row_end}), 列[{l8_col_start}, {l8_col_end})")
        logger.info(f"共同尺寸: {common_height} × {common_width}")
        
        # 裁剪到共同尺寸（从正确的位置裁剪）
        uav_bands_common = [band[uav_row_start:uav_row_start+common_height, uav_col_start:uav_col_start+common_width] 
                           for band in uav_bands_aligned]
        l8_bands_common = [band[l8_row_start:l8_row_start+common_height, l8_col_start:l8_col_start+common_width] 
                          for band in l8_bands_aligned]
        s2_bands_common = [band[:common_height, :common_width] for band in s2_bands_aligned]
        
        # 计算裁剪后的 transform（用于保存标准化影像）
        # 使用 rasterio.transform.xy 从行列号计算坐标
        import rasterio.transform
        common_ul_x, common_ul_y = rasterio.transform.xy(
            uav_aligned_transform, uav_row_start, uav_col_start, offset='ul'
        )
        # 创建裁剪后的 transform
        common_transform = rasterio.transform.Affine(
            uav_aligned_transform[0],  # pixel width
            uav_aligned_transform[1],  # rotation
            common_ul_x,                # upper left x
            uav_aligned_transform[3],   # rotation
            uav_aligned_transform[4],   # pixel height (negative)
            common_ul_y                 # upper left y
        )
        logger.info(f"共同区域 transform: {common_transform}")
        logger.info(f"共同区域左上角坐标: ({common_ul_x:.6f}, {common_ul_y:.6f})")
        
        # 展平为像元数组
        uav_pixels = np.column_stack([band.flatten() for band in uav_bands_common])
        l8_pixels = np.column_stack([band.flatten() for band in l8_bands_common])
        s2_pixels = np.column_stack([band.flatten() for band in s2_bands_common])
        
        n_pixels = len(uav_pixels)
        logger.info(f"总像元数: {n_pixels}")
        
        # 添加调试信息：检查数据统计
        logger.info("\n数据统计（过滤前）:")
        logger.info(f"  UAV: 形状={uav_pixels.shape}, NaN数量={np.isnan(uav_pixels).sum()}, 最小值={np.nanmin(uav_pixels):.2f}, 最大值={np.nanmax(uav_pixels):.2f}")
        logger.info(f"  L8:  形状={l8_pixels.shape}, NaN数量={np.isnan(l8_pixels).sum()}, 最小值={np.nanmin(l8_pixels):.2f}, 最大值={np.nanmax(l8_pixels):.2f}")
        logger.info(f"  S2:  形状={s2_pixels.shape}, NaN数量={np.isnan(s2_pixels).sum()}, 最小值={np.nanmin(s2_pixels):.2f}, 最大值={np.nanmax(s2_pixels):.2f}")
        
        # 创建有效掩码：同时检查S2、L8、UAV的nodata值
        valid_mask = np.ones(n_pixels, dtype=bool)
        
        # 过滤UAV的nodata和NaN
        uav_mask = np.ones(n_pixels, dtype=bool)
        if uav_nodata is not None:
            logger.info(f"  过滤UAV nodata值: {uav_nodata}")
            for i in range(uav_pixels.shape[1]):
                uav_mask = uav_mask & (uav_pixels[:, i] != uav_nodata) & (~np.isnan(uav_pixels[:, i]))
        else:
            logger.info("  过滤UAV NaN值")
            for i in range(uav_pixels.shape[1]):
                uav_mask = uav_mask & (~np.isnan(uav_pixels[:, i]))
        
        uav_valid_count = np.sum(uav_mask)
        logger.info(f"  UAV有效像元: {uav_valid_count} / {n_pixels} ({100 * uav_valid_count / n_pixels:.2f}%)")
        valid_mask = valid_mask & uav_mask
        
        # 过滤L8的nodata和NaN
        l8_mask = np.ones(n_pixels, dtype=bool)
        if l8_nodata is not None:
            logger.info(f"  过滤L8 nodata值: {l8_nodata}")
            for i in range(l8_pixels.shape[1]):
                l8_mask = l8_mask & (l8_pixels[:, i] != l8_nodata) & (~np.isnan(l8_pixels[:, i]))
        else:
            logger.info("  过滤L8 NaN值")
            for i in range(l8_pixels.shape[1]):
                l8_mask = l8_mask & (~np.isnan(l8_pixels[:, i]))
        
        l8_valid_count = np.sum(l8_mask)
        logger.info(f"  L8有效像元: {l8_valid_count} / {n_pixels} ({100 * l8_valid_count / n_pixels:.2f}%)")
        valid_mask = valid_mask & l8_mask
        
        # 过滤S2的nodata和NaN
        s2_mask = np.ones(n_pixels, dtype=bool)
        if s2_nodata is not None:
            logger.info(f"  过滤S2 nodata值: {s2_nodata}")
            for i in range(s2_pixels.shape[1]):
                s2_mask = s2_mask & (s2_pixels[:, i] != s2_nodata) & (~np.isnan(s2_pixels[:, i]))
        else:
            logger.info("  过滤S2 NaN值")
            for i in range(s2_pixels.shape[1]):
                s2_mask = s2_mask & (~np.isnan(s2_pixels[:, i]))
        
        s2_valid_count = np.sum(s2_mask)
        logger.info(f"  S2有效像元: {s2_valid_count} / {n_pixels} ({100 * s2_valid_count / n_pixels:.2f}%)")
        valid_mask = valid_mask & s2_mask
        
        # 提取有效像元
        uav_valid = uav_pixels[valid_mask]
        l8_valid = l8_pixels[valid_mask]
        s2_valid = s2_pixels[valid_mask]
        
        n_valid = len(uav_valid)
        logger.info(f"\n最终有效像元数（三者交集）: {n_valid} / {n_pixels} ({100 * n_valid / n_pixels:.2f}%)")
        
        if n_valid == 0:
            # 提供更详细的错误信息
            logger.error("\n详细诊断信息:")
            logger.error(f"  UAV单独有效: {uav_valid_count}")
            logger.error(f"  L8单独有效: {l8_valid_count}")
            logger.error(f"  S2单独有效: {s2_valid_count}")
            logger.error(f"  UAV+L8交集: {np.sum(uav_mask & l8_mask)}")
            logger.error(f"  UAV+S2交集: {np.sum(uav_mask & s2_mask)}")
            logger.error(f"  L8+S2交集: {np.sum(l8_mask & s2_mask)}")
            logger.error(f"  三者交集: {n_valid}")
            
            # 检查是否有任何两个的交集
            if np.sum(uav_mask & l8_mask) > 0:
                logger.warning("  UAV和L8有重叠，但S2可能没有覆盖该区域")
            if np.sum(uav_mask & s2_mask) > 0:
                logger.warning("  UAV和S2有重叠，但L8可能没有覆盖该区域")
            if np.sum(l8_mask & s2_mask) > 0:
                logger.warning("  L8和S2有重叠，但UAV可能没有覆盖该区域")
            
            raise ValueError(
                "没有找到同时有S2、L8、UAV有效值的像元！\n"
                "可能的原因：\n"
                "1. 对齐后的影像在重叠区域包含NaN值（对齐时未覆盖的区域）\n"
                "2. nodata值设置不正确\n"
                "3. 三个影像的实际重叠区域很小\n"
                "请检查对齐后的影像，确认重叠区域是否有有效值。"
            )
        
        # 5. 生成配对数据：每个像元位置生成两个样本（S2和L8）
        logger.info("\n生成配对数据...")
        
        # 确定统一的波段数量（应该是4个：G, R, REG, NIR）
        # S2有4个波段，L8有3个波段（缺少REG），需要统一为4个
        n_sat_bands_unified = 4  # G, R, REG, NIR
        n_s2_bands = s2_valid.shape[1]
        n_l8_bands = l8_valid.shape[1]
        n_uav_bands = uav_valid.shape[1]
        
        logger.info(f"波段数量: S2={n_s2_bands}, L8={n_l8_bands}, UAV={n_uav_bands}, 统一为={n_sat_bands_unified}")
        
        for i in range(n_valid):
            # S2样本：(S2光谱, sensor_id=0, UAV光谱)
            s2_bands = s2_valid[i].tolist()
            # 确保S2有4个波段
            if len(s2_bands) < n_sat_bands_unified:
                s2_bands.extend([0.0] * (n_sat_bands_unified - len(s2_bands)))
            elif len(s2_bands) > n_sat_bands_unified:
                s2_bands = s2_bands[:n_sat_bands_unified]
            
            pair_s2 = {
                'satellite_bands': s2_bands,  # 卫星观测光谱（S2）- 4个波段
                'sensor_id': sensor_labels['S2'],  # 传感器ID = 0
                'uav_bands': uav_valid[i].tolist()  # UAV本征光谱真值（聚合到S2后的值）
            }
            all_pairs.append(pair_s2)
            
            # L8样本：(L8光谱, sensor_id=1, UAV光谱)
            l8_bands = l8_valid[i].tolist()
            # L8只有3个波段（G, R, NIR），需要补充REG波段为0
            # 假设L8的顺序是G, R, NIR（需要根据实际配置调整）
            if len(l8_bands) == 3:
                # 补充REG波段为0，顺序变为：G, R, REG=0, NIR
                l8_bands_unified = [l8_bands[0], l8_bands[1], 0.0, l8_bands[2]]
            elif len(l8_bands) == 4:
                l8_bands_unified = l8_bands
            else:
                # 如果波段数不对，填充或截断
                l8_bands_unified = l8_bands[:3] + [0.0] if len(l8_bands) >= 3 else l8_bands + [0.0] * (4 - len(l8_bands))
                if len(l8_bands_unified) < 4:
                    l8_bands_unified.extend([0.0] * (4 - len(l8_bands_unified)))
            
            pair_l8 = {
                'satellite_bands': l8_bands_unified,  # 卫星观测光谱（L8）- 统一为4个波段（REG=0）
                'sensor_id': sensor_labels['L8'],  # 传感器ID = 1
                'uav_bands': uav_valid[i].tolist()  # UAV本征光谱真值（聚合到S2后的值）
            }
            all_pairs.append(pair_l8)
        
        logger.info(f"生成配对数据: {n_valid} 个像元位置 × 2 个传感器 = {len(all_pairs)} 个样本")
        
        # 4. 构建DataFrame
        if not all_pairs:
            raise ValueError("没有生成任何配对数据！请检查影像路径和重叠区域。")
        
        # 将列表转换为DataFrame，每个波段作为单独的列
        # 现在所有样本都有统一的波段数量
        n_sat_bands = n_sat_bands_unified
        n_uav_bands = len(all_pairs[0]['uav_bands'])
        
        data_dict = {}
        # 卫星波段列（统一为4个：G, R, REG, NIR）
        for i in range(n_sat_bands):
            data_dict[f'SAT_band_{i+1}'] = [pair['satellite_bands'][i] for pair in all_pairs]
        # 传感器ID列
        data_dict['sensor_id'] = [pair['sensor_id'] for pair in all_pairs]
        # UAV波段列
        for i in range(n_uav_bands):
            data_dict[f'UAV_band_{i+1}'] = [pair['uav_bands'][i] for pair in all_pairs]
        
        df = pd.DataFrame(data_dict)
        
        # 保存原始配对数据表
        if output_csv:
            logger.info(f"保存原始配对数据表: {output_csv}")
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_csv, index=False, encoding='utf-8-sig')
            logger.info(f"  已保存 {len(df)} 个配对样本")
        
        # 6. 标准化处理（Z-score标准化）
        df_normalized = None  # 初始化，如果进行了标准化，会被赋值
        if normalize:
            logger.info("\n" + "=" * 60)
            logger.info("标准化处理 (Z-score: (x - mean) / std)")
            logger.info("=" * 60)
            
            from .normalization import StandardScaler
            
            # 强制使用StandardScaler（Z-score标准化）
            logger.info("使用 StandardScaler (Z-score标准化)")
            
            # 创建三个独立的标准化器
            scaler_s2 = StandardScaler()
            scaler_l8 = StandardScaler()
            scaler_uav = StandardScaler()
            
            # 分离S2和L8的数据
            s2_mask = df['sensor_id'] == sensor_labels['S2']
            l8_mask = df['sensor_id'] == sensor_labels['L8']
            
            # 提取波段数据
            s2_band_cols = [f'SAT_band_{i+1}' for i in range(n_sat_bands)]
            l8_band_cols = [f'SAT_band_{i+1}' for i in range(n_sat_bands)]
            uav_band_cols = [f'UAV_band_{i+1}' for i in range(n_uav_bands)]
            
            s2_data = df[s2_mask][s2_band_cols].values
            l8_data = df[l8_mask][l8_band_cols].values
            uav_data = df[uav_band_cols].values
            
            logger.info(f"S2数据形状: {s2_data.shape}")
            logger.info(f"L8数据形状: {l8_data.shape}")
            logger.info(f"UAV数据形状: {uav_data.shape}")
            
            # 拟合标准化器
            logger.info("\n拟合标准化器...")
            # S2: 所有4个波段都参与标准化
            scaler_s2.fit(s2_data)
            # UAV: 所有波段都参与标准化
            scaler_uav.fit(uav_data)
            
            # L8: REG波段（索引2）不参与标准化，只对G(0)、R(1)、NIR(3)计算均值和标准差
            logger.info("L8标准化: REG波段（索引2，掩码为0）不参与标准化计算")
            
            # 对每个波段独立计算均值和标准差（排除REG波段）
            l8_mean = np.full((1, n_sat_bands), np.nan)
            l8_std = np.full((1, n_sat_bands), np.nan)
            
            for band_idx in range(n_sat_bands):
                if band_idx == 2:  # REG波段
                    # REG波段：mean=0, std=1（不标准化，保持原值0）
                    l8_mean[0, band_idx] = 0.0
                    l8_std[0, band_idx] = 1.0
                    logger.info(f"  L8波段{band_idx+1} (REG): mean=0.0, std=1.0 (掩码，不参与标准化)")
                else:
                    # 其他波段：正常计算均值和标准差
                    band_data = l8_data[:, band_idx]
                    valid_data = band_data[~np.isnan(band_data)]
                    if len(valid_data) > 0:
                        l8_mean[0, band_idx] = np.mean(valid_data)
                        l8_std[0, band_idx] = np.std(valid_data)
                        if l8_std[0, band_idx] == 0:
                            l8_std[0, band_idx] = 1.0
                        logger.info(f"  L8波段{band_idx+1}: mean={l8_mean[0, band_idx]:.6f}, std={l8_std[0, band_idx]:.6f}")
                    else:
                        l8_mean[0, band_idx] = 0.0
                        l8_std[0, band_idx] = 1.0
            
            # 手动设置L8标准化器的参数
            scaler_l8.mean_ = l8_mean
            scaler_l8.std_ = l8_std
            scaler_l8.scale_ = 1.0 / l8_std
            scaler_l8.fitted = True
            
            # 记录标准化器参数
            logger.info("\n标准化器参数:")
            logger.info(f"  S2: mean={scaler_s2.mean_.flatten()}, std={scaler_s2.std_.flatten()}")
            logger.info(f"  L8: mean={scaler_l8.mean_.flatten()}, std={scaler_l8.std_.flatten()}")
            logger.info(f"  UAV: mean={scaler_uav.mean_.flatten()}, std={scaler_uav.std_.flatten()}")
            
            # 保存标准化器参数
            if output_scalers_dir:
                output_scalers_dir = Path(output_scalers_dir)
                output_scalers_dir.mkdir(parents=True, exist_ok=True)
                
                scaler_s2.save(output_scalers_dir / 'scaler_s2.pkl')
                scaler_l8.save(output_scalers_dir / 'scaler_l8.pkl')
                scaler_uav.save(output_scalers_dir / 'scaler_uav.pkl')
                
                logger.info(f"标准化器参数已保存到: {output_scalers_dir}")
            
            # 标准化数据
            logger.info("\n标准化数据...")
            s2_normalized = scaler_s2.transform(s2_data)
            # L8标准化：REG波段（索引2）保持为0，不进行标准化
            l8_normalized = scaler_l8.transform(l8_data)
            # 确保REG波段（索引2）保持为0
            l8_normalized[:, 2] = 0.0
            logger.info("  L8 REG波段（索引2）保持为0，未标准化")
            uav_normalized = scaler_uav.transform(uav_data)
            
            # 创建标准化后的DataFrame
            df_normalized = df.copy()
            
            # 更新S2数据
            for i, col in enumerate(s2_band_cols):
                df_normalized.loc[s2_mask, col] = s2_normalized[:, i]
            
            # 更新L8数据
            for i, col in enumerate(l8_band_cols):
                df_normalized.loc[l8_mask, col] = l8_normalized[:, i]
            
            # 更新UAV数据
            for i, col in enumerate(uav_band_cols):
                df_normalized[col] = uav_normalized[:, i]
            
            # 保存标准化后的配对数据表
            if output_normalized_csv:
                logger.info(f"保存标准化后的配对数据表: {output_normalized_csv}")
                output_normalized_csv.parent.mkdir(parents=True, exist_ok=True)
                df_normalized.to_csv(output_normalized_csv, index=False, encoding='utf-8-sig')
                logger.info(f"  已保存 {len(df_normalized)} 个标准化配对样本")
            
            # 生成分布图
            if output_distribution_plots_dir:
                logger.info("\n生成标准化前后分布图...")
                output_distribution_plots_dir = Path(output_distribution_plots_dir)
                output_distribution_plots_dir.mkdir(parents=True, exist_ok=True)
                
                # 波段名称
                band_names = ['G', 'R', 'REG', 'NIR']
                
                # 绘制S2分布图
                logger.info("  绘制S2分布图...")
                _plot_distributions(
                    data_before=s2_data,
                    data_after=s2_normalized,
                    sensor_name='S2',
                    band_names=band_names,
                    scaler=scaler_s2,
                    output_dir=output_distribution_plots_dir
                )
                
                # 绘制L8分布图
                logger.info("  绘制L8分布图...")
                _plot_distributions(
                    data_before=l8_data,
                    data_after=l8_normalized,
                    sensor_name='L8',
                    band_names=band_names,
                    scaler=scaler_l8,
                    output_dir=output_distribution_plots_dir
                )
                
                # 绘制UAV分布图
                logger.info("  绘制UAV分布图...")
                _plot_distributions(
                    data_before=uav_data,
                    data_after=uav_normalized,
                    sensor_name='UAV',
                    band_names=band_names[:n_uav_bands],
                    scaler=scaler_uav,
                    output_dir=output_distribution_plots_dir
                )
                
                logger.info(f"  分布图已保存到: {output_distribution_plots_dir}")
            
            # 标准化对齐后的影像
            logger.info("\n标准化对齐后的影像...")
            
            # 辅助函数：标准化单个波段
            def normalize_band(band_array, scaler, band_idx):
                """标准化单个波段"""
                band_flat = band_array.flatten()
                valid_mask = ~np.isnan(band_flat)
                band_normalized = band_flat.copy()
                
                if valid_mask.any():
                    if hasattr(scaler, 'mean_'):
                        # StandardScaler
                        if band_idx < scaler.mean_.shape[1]:
                            band_mean = scaler.mean_[0, band_idx]
                            band_std = scaler.std_[0, band_idx]
                        else:
                            band_mean = scaler.mean_[0, -1]
                            band_std = scaler.std_[0, -1]
                        band_normalized[valid_mask] = (band_flat[valid_mask] - band_mean) / band_std
                    else:
                        # MinMaxScaler
                        if band_idx < scaler.min_.shape[1]:
                            band_min = scaler.min_[0, band_idx]
                            band_scale = scaler.scale_[0, band_idx]
                        else:
                            band_min = scaler.min_[0, -1]
                            band_scale = scaler.scale_[0, -1]
                        band_normalized[valid_mask] = (band_flat[valid_mask] - band_min) * band_scale + scaler.feature_range[0]
                
                return band_normalized.reshape(band_array.shape)
            
            # 标准化S2对齐影像
            if output_normalized_s2_tif:
                logger.info(f"标准化并保存S2影像: {output_normalized_s2_tif}")
                s2_bands_normalized = []
                for i, band in enumerate(s2_bands_common):
                    s2_bands_normalized.append(normalize_band(band, scaler_s2, i))
                
                save_aligned_raster(
                    aligned_bands=s2_bands_normalized,
                    output_path=output_normalized_s2_tif,
                    transform=common_transform,
                    crs=s2_crs,
                    dtype="float32",
                    nodata=None
                )
            
            # 标准化UAV对齐影像
            if output_normalized_uav_tif:
                logger.info(f"标准化并保存UAV影像: {output_normalized_uav_tif}")
                uav_bands_normalized = []
                for i, band in enumerate(uav_bands_common):
                    uav_bands_normalized.append(normalize_band(band, scaler_uav, i))
                
                save_aligned_raster(
                    aligned_bands=uav_bands_normalized,
                    output_path=output_normalized_uav_tif,
                    transform=common_transform,
                    crs=s2_crs,
                    dtype="float32",
                    nodata=None
                )
            
            # 标准化L8对齐影像
            if output_normalized_l8_tif:
                logger.info(f"标准化并保存L8影像: {output_normalized_l8_tif}")
                l8_bands_normalized = []
                for i, band in enumerate(l8_bands_common):
                    # L8只有3个波段（G, R, NIR），但标准化器拟合了4个（G, R, REG=0, NIR）
                    # 映射：L8波段0→标准化器0(G), 1→1(R), 2→3(NIR)
                    # 注意：L8的REG波段（索引2）在标准化器中对应索引2，但应该保持为0
                    if i == 0:
                        band_idx = 0  # G
                    elif i == 1:
                        band_idx = 1  # R
                    elif i == 2:
                        # L8的第三个波段是NIR，对应标准化器的索引3
                        band_idx = 3  # NIR（跳过REG=2）
                    else:
                        band_idx = min(i, scaler_l8.mean_.shape[1] - 1)
                    
                    # 标准化波段
                    normalized_band = normalize_band(band, scaler_l8, band_idx)
                    l8_bands_normalized.append(normalized_band)
                
                # 注意：L8只有3个波段（G, R, NIR），但标准化后的影像应该有4个波段
                # 需要在索引2位置插入REG波段（全0）
                # 但这里l8_bands_common只有3个波段，所以不需要插入
                # 如果后续需要4个波段，可以在这里添加
                logger.info(f"  L8标准化影像: {len(l8_bands_normalized)} 个波段 (G, R, NIR)")
                
                save_aligned_raster(
                    aligned_bands=l8_bands_normalized,
                    output_path=output_normalized_l8_tif,
                    transform=common_transform,
                    crs=s2_crs,
                    dtype="float32",
                    nodata=None
                )
            
            logger.info("标准化处理完成！")
        
        logger.info("\n配对数据统计:")
        logger.info(f"  总样本数: {len(df)}")
        logger.info(f"  S2样本数: {len(df[df['sensor_id'] == sensor_labels['S2']])}")
        if l8_raster_path and l8_raster_path.exists():
            logger.info(f"  L8样本数: {len(df[df['sensor_id'] == sensor_labels['L8']])}")
        logger.info(f"  卫星波段数: {n_sat_bands}")
        logger.info(f"  UAV波段数: {n_uav_bands}")
        
        # 如果进行了标准化，返回标准化后的DataFrame；否则返回原始DataFrame
        if df_normalized is not None:
            logger.info("\n返回标准化后的DataFrame（用于后续训练）")
            return df_normalized
        else:
            logger.info("\n返回原始DataFrame（未标准化）")
            return df

