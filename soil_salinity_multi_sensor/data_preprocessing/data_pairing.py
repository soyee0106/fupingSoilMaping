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
from typing import Dict, List, Optional, Tuple
import logging

from .band_matching import create_band_mapping
from .spectral_indices import calculate_all_indices

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
            transform = src.transform
            
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

