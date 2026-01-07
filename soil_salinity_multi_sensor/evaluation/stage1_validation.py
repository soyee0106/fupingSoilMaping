"""
Stage 1 验证实验模块

实现两个验证实验：
1. 基础映射精度：计算S2和L8预测光谱与真实UAV光谱的R²和RMSE
2. 跨传感器一致性：计算同一位置S2和L8预测光谱的差异
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging
from tqdm import tqdm
import rasterio
from rasterio.transform import Affine

from models.stage1_decoder import SensorBiasDecoder
from data_preprocessing.spectral_indices import calculate_all_indices
from data_preprocessing.normalization_utils import load_scalers
from evaluation.metrics import calculate_r2, calculate_rmse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_stage1_model(
    model_path: Path,
    model_config: Dict,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> SensorBiasDecoder:
    """
    加载训练好的Stage 1模型。
    
    参数:
        model_path: 模型权重文件路径
        model_config: 模型配置字典
        device: 设备（'cuda'或'cpu'）
    
    返回:
        加载好的模型
    """
    logger.info(f"Loading Stage 1 model from {model_path}")
    
    stage1_config = model_config.get('stage1', {})
    model = SensorBiasDecoder(**stage1_config)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    logger.info("Model loaded successfully")
    return model


def predict_uav_spectrum(
    model: SensorBiasDecoder,
    satellite_bands: np.ndarray,
    spectral_indices: np.ndarray,
    sensor_id: int,
    band_mask: np.ndarray,
    scaler_satellite: Optional[object] = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> np.ndarray:
    """
    使用Stage 1模型预测UAV光谱。
    
    参数:
        model: Stage 1模型
        satellite_bands: 卫星波段 (4个: G, R, REG, NIR)
        spectral_indices: 光谱指数 (26个)
        sensor_id: 传感器ID (0=S2, 1=L8)
        band_mask: 波段掩码 (4个)
        scaler_satellite: 卫星波段标准化器（如果数据未标准化）
        device: 设备
    
    返回:
        预测的UAV波段 (4个: G, R, REG, NIR)
    """
    # 标准化卫星波段（如果提供了scaler且数据未标准化）
    if scaler_satellite is not None:
        # 检查数据是否已标准化（通过范围判断）
        if np.abs(satellite_bands).max() > 10:
            satellite_bands = scaler_satellite.transform(satellite_bands.reshape(1, -1)).flatten()
    
    # 转换为tensor
    satellite_bands_t = torch.FloatTensor(satellite_bands).unsqueeze(0).to(device)
    spectral_indices_t = torch.FloatTensor(spectral_indices).unsqueeze(0).to(device)
    
    # 传感器独热编码
    sensor_onehot = torch.zeros(1, 2, dtype=torch.float32).to(device)
    sensor_onehot[0, sensor_id] = 1.0
    
    # 波段掩码
    band_mask_t = torch.FloatTensor(band_mask).unsqueeze(0).to(device)
    
    # 预测
    with torch.no_grad():
        uav_bands_pred = model(
            satellite_bands_t,
            spectral_indices_t,
            sensor_onehot,
            band_mask_t
        )
    
    return uav_bands_pred.cpu().numpy().flatten()


def experiment1_basic_mapping_accuracy(
    model: SensorBiasDecoder,
    val_df: pd.DataFrame,
    scaler_s2: Optional[object] = None,
    scaler_l8: Optional[object] = None,
    scaler_uav: Optional[object] = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict[str, Dict[str, float]]:
    """
    实验1：基础映射精度评估
    
    在验证集上计算S2和L8预测光谱与真实UAV光谱的R²和RMSE。
    
    参数:
        model: Stage 1模型
        val_df: 验证集DataFrame（包含SAT_band_*, UAV_band_*, sensor_id列）
        scaler_satellite: 卫星波段标准化器
        scaler_uav: UAV波段标准化器（用于反标准化）
        device: 设备
    
    返回:
        包含S2和L8评估指标的字典
    """
    logger.info("=" * 60)
    logger.info("实验1：基础映射精度评估")
    logger.info("=" * 60)
    
    # 提取列名
    sat_band_cols = [col for col in val_df.columns if col.startswith('SAT_band_')]
    uav_band_cols = [col for col in val_df.columns if col.startswith('UAV_band_')]
    sat_band_cols.sort()
    uav_band_cols.sort()
    
    # 分离S2和L8数据
    s2_mask = val_df['sensor_id'] == 0
    l8_mask = val_df['sensor_id'] == 1
    
    s2_df = val_df[s2_mask].copy()
    l8_df = val_df[l8_mask].copy()
    
    logger.info(f"S2样本数: {len(s2_df)}")
    logger.info(f"L8样本数: {len(l8_df)}")
    
    results = {}
    
    # 评估S2
    if len(s2_df) > 0:
        logger.info("\n评估S2预测精度...")
        s2_preds = []
        s2_trues = []
        
        for idx, row in tqdm(s2_df.iterrows(), total=len(s2_df), desc="S2预测"):
            # 提取输入
            satellite_bands = row[sat_band_cols].values.astype(np.float32)
            uav_true = row[uav_band_cols].values.astype(np.float32)
            
            # 计算光谱指数
            band_dict = {
                'G': satellite_bands[0],
                'R': satellite_bands[1],
                'REG': satellite_bands[2],
                'NIR': satellite_bands[3]
            }
            indices_dict = calculate_all_indices(band_dict)
            index_order = [
                'S1', 'S1REG', 'NDSI', 'NDSIREG', 'SI1', 'SI1REG', 'SI2', 'SI2REG',
                'SI3', 'SI3REG', 'SIT', 'SITREG', 'Int1', 'Int1REG', 'Int2', 'Int2REG',
                'NDVI', 'NDVIREG', 'SAVI', 'SAVIREG', 'BI', 'BIREG', 'IFe2O3', 'IFe2O3REG',
                'DVI', 'DVIREG'
            ]
            spectral_indices = np.array([indices_dict[idx] for idx in index_order], dtype=np.float32)
            spectral_indices = np.nan_to_num(spectral_indices, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 预测
            band_mask = np.array([1.0, 1.0, 1.0, 1.0])  # S2所有波段有效
            uav_pred = predict_uav_spectrum(
                model, satellite_bands, spectral_indices, 0, band_mask,
                scaler_s2, device
            )
            
            # 反标准化（如果需要）
            if scaler_uav is not None:
                uav_pred = scaler_uav.inverse_transform(uav_pred.reshape(1, -1)).flatten()
                uav_true = scaler_uav.inverse_transform(uav_true.reshape(1, -1)).flatten()
            
            s2_preds.append(uav_pred)
            s2_trues.append(uav_true)
        
        s2_preds = np.array(s2_preds)
        s2_trues = np.array(s2_trues)
        
        # 计算指标（所有波段一起计算）
        s2_r2 = calculate_r2(s2_trues.flatten(), s2_preds.flatten())
        s2_rmse = calculate_rmse(s2_trues.flatten(), s2_preds.flatten())
        
        # 每个波段的指标
        s2_r2_per_band = [calculate_r2(s2_trues[:, i], s2_preds[:, i]) for i in range(4)]
        s2_rmse_per_band = [calculate_rmse(s2_trues[:, i], s2_preds[:, i]) for i in range(4)]
        
        results['S2'] = {
            'R2_overall': s2_r2,
            'RMSE_overall': s2_rmse,
            'R2_per_band': s2_r2_per_band,
            'RMSE_per_band': s2_rmse_per_band
        }
        
        logger.info(f"S2总体R²: {s2_r2:.4f}, RMSE: {s2_rmse:.4f}")
        logger.info(f"S2各波段R²: {s2_r2_per_band}")
        logger.info(f"S2各波段RMSE: {s2_rmse_per_band}")
    
    # 评估L8
    if len(l8_df) > 0:
        logger.info("\n评估L8预测精度...")
        l8_preds = []
        l8_trues = []
        
        for idx, row in tqdm(l8_df.iterrows(), total=len(l8_df), desc="L8预测"):
            # 提取输入
            satellite_bands = row[sat_band_cols].values.astype(np.float32)
            uav_true = row[uav_band_cols].values.astype(np.float32)
            
            # 计算光谱指数
            band_dict = {
                'G': satellite_bands[0],
                'R': satellite_bands[1],
                'REG': satellite_bands[2],  # L8的REG是0，但计算指数时仍使用
                'NIR': satellite_bands[3]
            }
            indices_dict = calculate_all_indices(band_dict)
            index_order = [
                'S1', 'S1REG', 'NDSI', 'NDSIREG', 'SI1', 'SI1REG', 'SI2', 'SI2REG',
                'SI3', 'SI3REG', 'SIT', 'SITREG', 'Int1', 'Int1REG', 'Int2', 'Int2REG',
                'NDVI', 'NDVIREG', 'SAVI', 'SAVIREG', 'BI', 'BIREG', 'IFe2O3', 'IFe2O3REG',
                'DVI', 'DVIREG'
            ]
            spectral_indices = np.array([indices_dict[idx] for idx in index_order], dtype=np.float32)
            spectral_indices = np.nan_to_num(spectral_indices, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 预测
            band_mask = np.array([1.0, 1.0, 0.0, 1.0])  # L8的REG缺失
            uav_pred = predict_uav_spectrum(
                model, satellite_bands, spectral_indices, 1, band_mask,
                scaler_l8, device
            )
            
            # 反标准化（如果需要）
            if scaler_uav is not None:
                uav_pred = scaler_uav.inverse_transform(uav_pred.reshape(1, -1)).flatten()
                uav_true = scaler_uav.inverse_transform(uav_true.reshape(1, -1)).flatten()
            
            l8_preds.append(uav_pred)
            l8_trues.append(uav_true)
        
        l8_preds = np.array(l8_preds)
        l8_trues = np.array(l8_trues)
        
        # 计算指标
        l8_r2 = calculate_r2(l8_trues.flatten(), l8_preds.flatten())
        l8_rmse = calculate_rmse(l8_trues.flatten(), l8_preds.flatten())
        
        # 每个波段的指标
        l8_r2_per_band = [calculate_r2(l8_trues[:, i], l8_preds[:, i]) for i in range(4)]
        l8_rmse_per_band = [calculate_rmse(l8_trues[:, i], l8_preds[:, i]) for i in range(4)]
        
        results['L8'] = {
            'R2_overall': float(l8_r2),
            'RMSE_overall': float(l8_rmse),
            'R2_per_band': [float(x) for x in l8_r2_per_band],
            'RMSE_per_band': [float(x) for x in l8_rmse_per_band]
        }
        
        logger.info(f"L8总体R²: {l8_r2:.4f}, RMSE: {l8_rmse:.4f}")
        logger.info(f"L8各波段R²: {l8_r2_per_band}")
        logger.info(f"L8各波段RMSE: {l8_rmse_per_band}")
    
    return results


def experiment2_cross_sensor_consistency(
    model: SensorBiasDecoder,
    val_df: pd.DataFrame,
    scaler_s2: Optional[object] = None,
    scaler_l8: Optional[object] = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict[str, float]:
    """
    实验2：跨传感器一致性评估
    
    选取同一地面位置的S2和L8样本，分别输入模型，得到两个预测光谱。
    计算这两个预测光谱的差异（欧氏距离）。
    
    参数:
        model: Stage 1模型
        val_df: 验证集DataFrame（需要包含位置信息或能匹配同一位置的S2和L8样本）
        scaler_satellite: 卫星波段标准化器
        device: 设备
    
    返回:
        包含一致性指标的字典
    """
    logger.info("=" * 60)
    logger.info("实验2：跨传感器一致性评估")
    logger.info("=" * 60)
    
    # 提取列名
    sat_band_cols = [col for col in val_df.columns if col.startswith('SAT_band_')]
    sat_band_cols.sort()
    
    # 分离S2和L8数据
    s2_df = val_df[val_df['sensor_id'] == 0].copy().reset_index(drop=True)
    l8_df = val_df[val_df['sensor_id'] == 1].copy().reset_index(drop=True)
    
    logger.info(f"S2样本数: {len(s2_df)}")
    logger.info(f"L8样本数: {len(l8_df)}")
    
    # 通过UAV波段值匹配同一位置的样本
    # 同一位置的S2和L8样本应该有相同的UAV波段值（真值）
    uav_band_cols = [col for col in val_df.columns if col.startswith('UAV_band_')]
    uav_band_cols.sort()
    
    # 创建匹配字典：UAV波段值 -> (s2_index, l8_index)
    s2_uav_values = s2_df[uav_band_cols].values
    l8_uav_values = l8_df[uav_band_cols].values
    
    # 使用欧氏距离匹配（允许小的数值误差）
    matched_pairs = []
    tolerance = 1e-5  # 允许的数值误差
    
    for s2_idx, s2_uav in enumerate(s2_uav_values):
        # 找到最接近的L8样本
        distances = np.linalg.norm(l8_uav_values - s2_uav, axis=1)
        min_dist_idx = np.argmin(distances)
        min_dist = distances[min_dist_idx]
        
        if min_dist < tolerance:
            matched_pairs.append((s2_idx, min_dist_idx))
    
    if len(matched_pairs) == 0:
        logger.warning("没有找到匹配的S2和L8样本对")
        logger.warning("尝试使用索引匹配（假设数据按位置顺序排列）")
        # 回退到索引匹配
        min_len = min(len(s2_df), len(l8_df))
        matched_pairs = [(i, i) for i in range(min_len)]
    
    logger.info(f"匹配到 {len(matched_pairs)} 对S2-L8样本")
    
    s2_preds = []
    l8_preds = []
    euclidean_distances = []
    
    for s2_idx, l8_idx in tqdm(matched_pairs, desc="计算跨传感器一致性"):
        # S2预测
        s2_row = s2_df.iloc[s2_idx]
        s2_satellite_bands = s2_row[sat_band_cols].values.astype(np.float32)
        
        band_dict = {
            'G': s2_satellite_bands[0],
            'R': s2_satellite_bands[1],
            'REG': s2_satellite_bands[2],
            'NIR': s2_satellite_bands[3]
        }
        indices_dict = calculate_all_indices(band_dict)
        index_order = [
            'S1', 'S1REG', 'NDSI', 'NDSIREG', 'SI1', 'SI1REG', 'SI2', 'SI2REG',
            'SI3', 'SI3REG', 'SIT', 'SITREG', 'Int1', 'Int1REG', 'Int2', 'Int2REG',
            'NDVI', 'NDVIREG', 'SAVI', 'SAVIREG', 'BI', 'BIREG', 'IFe2O3', 'IFe2O3REG',
            'DVI', 'DVIREG'
        ]
        s2_spectral_indices = np.array([indices_dict[idx] for idx in index_order], dtype=np.float32)
        s2_spectral_indices = np.nan_to_num(s2_spectral_indices, nan=0.0, posinf=0.0, neginf=0.0)
        
        s2_band_mask = np.array([1.0, 1.0, 1.0, 1.0])
        s2_pred = predict_uav_spectrum(
            model, s2_satellite_bands, s2_spectral_indices, 0, s2_band_mask,
            scaler_s2, device
        )
        
        # L8预测
        l8_row = l8_df.iloc[l8_idx]
        l8_satellite_bands = l8_row[sat_band_cols].values.astype(np.float32)
        
        band_dict = {
            'G': l8_satellite_bands[0],
            'R': l8_satellite_bands[1],
            'REG': l8_satellite_bands[2],  # L8的REG是0
            'NIR': l8_satellite_bands[3]
        }
        indices_dict = calculate_all_indices(band_dict)
        l8_spectral_indices = np.array([indices_dict[idx] for idx in index_order], dtype=np.float32)
        l8_spectral_indices = np.nan_to_num(l8_spectral_indices, nan=0.0, posinf=0.0, neginf=0.0)
        
        l8_band_mask = np.array([1.0, 1.0, 0.0, 1.0])
        l8_pred = predict_uav_spectrum(
            model, l8_satellite_bands, l8_spectral_indices, 1, l8_band_mask,
            scaler_l8, device
        )
        
        s2_preds.append(s2_pred)
        l8_preds.append(l8_pred)
        
        # 计算欧氏距离
        euclidean_dist = np.linalg.norm(s2_pred - l8_pred)
        euclidean_distances.append(euclidean_dist)
    
    s2_preds = np.array(s2_preds)
    l8_preds = np.array(l8_preds)
    euclidean_distances = np.array(euclidean_distances)
    
    # 计算统计指标
    mean_distance = np.mean(euclidean_distances)
    std_distance = np.std(euclidean_distances)
    median_distance = np.median(euclidean_distances)
    
    # 计算每个波段的平均差异
    band_differences = np.abs(s2_preds - l8_preds)
    mean_diff_per_band = np.mean(band_differences, axis=0)
    
    results = {
        'mean_euclidean_distance': float(mean_distance),
        'std_euclidean_distance': float(std_distance),
        'median_euclidean_distance': float(median_distance),
        'mean_absolute_diff_per_band': [float(x) for x in mean_diff_per_band]
    }
    
    logger.info(f"平均欧氏距离: {mean_distance:.4f} ± {std_distance:.4f}")
    logger.info(f"中位数欧氏距离: {median_distance:.4f}")
    logger.info(f"各波段平均绝对差异: {mean_diff_per_band}")
    
    return results

