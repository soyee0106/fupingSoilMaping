"""
Stage 1 推理模块

使用训练好的Stage 1模型对整个影像进行预测，输出校正后的UAV光谱影像。
"""

import torch
import numpy as np
import rasterio
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging
from tqdm import tqdm

from models.stage1_decoder import SensorBiasDecoder
from data_preprocessing.spectral_indices import calculate_all_indices
from data_preprocessing.data_pairing import DataPairer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def predict_raster(
    model: SensorBiasDecoder,
    raster_path: Path,
    band_mapping_config: Dict,
    sensor_id: int,
    output_path: Path,
    scaler_satellite: Optional[object] = None,
    scaler_uav: Optional[object] = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    batch_size: int = 1000
) -> None:
    """
    对整个影像进行预测，输出校正后的UAV光谱影像。
    
    参数:
        model: Stage 1模型
        raster_path: 输入卫星影像路径
        band_mapping_config: 波段映射配置
        sensor_id: 传感器ID (0=S2, 1=L8)
        output_path: 输出影像路径
        scaler_satellite: 卫星波段标准化器
        scaler_uav: UAV波段标准化器（用于反标准化）
        device: 设备
        batch_size: 批处理大小
    """
    logger.info("=" * 60)
    logger.info("Stage 1 影像推理")
    logger.info("=" * 60)
    logger.info(f"输入影像: {raster_path}")
    logger.info(f"输出影像: {output_path}")
    logger.info(f"传感器ID: {sensor_id} ({'S2' if sensor_id == 0 else 'L8'})")
    
    # 读取影像
    with rasterio.open(raster_path) as src:
        height = src.height
        width = src.width
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
        
        logger.info(f"影像尺寸: {height} × {width}")
        logger.info(f"波段数: {src.count}")
        
        # 根据传感器类型选择波段映射
        if sensor_id == 0:  # S2
            sensor_config = band_mapping_config.get('s2', {})
            band_mask = np.array([1.0, 1.0, 1.0, 1.0])  # S2所有波段有效
        else:  # L8
            sensor_config = band_mapping_config.get('l8', {})
            band_mask = np.array([1.0, 1.0, 0.0, 1.0])  # L8的REG缺失
        
        # 查找波段索引
        pairer = DataPairer()
        band_indices = pairer.find_bands_by_mapping(raster_path, sensor_config.get('band_mapping', {}))
        logger.info(f"波段索引: {band_indices}")
        
        # 读取波段数据（G, R, REG, NIR）
        bands_data = []
        band_names = ['G', 'R', 'REG', 'NIR']
        for i, band_name in enumerate(band_names):
            if band_name == 'REG' and sensor_id == 1:
                # L8没有REG波段，填充0
                bands_data.append(np.zeros((height, width), dtype=np.float32))
            else:
                band_idx = band_indices.get(band_name)
                if band_idx is not None:
                    band_data = src.read(band_idx).astype(np.float32)
                    if nodata is not None:
                        band_data = np.where(band_data == nodata, np.nan, band_data)
                    bands_data.append(band_data)
                else:
                    bands_data.append(np.zeros((height, width), dtype=np.float32))
        
        # 转换为 (height, width, bands) 格式
        satellite_bands = np.stack(bands_data, axis=2)  # (H, W, 4)
        
        # 创建输出数组
        uav_bands = np.zeros((height, width, 4), dtype=np.float32)
        uav_bands.fill(np.nan)
        
        # 批处理预测
        logger.info("开始预测...")
        valid_mask = ~np.isnan(satellite_bands).any(axis=2)
        valid_pixels = np.sum(valid_mask)
        logger.info(f"有效像元数: {valid_pixels} / {height * width}")
        
        # 获取有效像素的坐标
        valid_coords = np.where(valid_mask)
        n_valid = len(valid_coords[0])
        
        # 分批处理
        for batch_start in tqdm(range(0, n_valid, batch_size), desc="预测进度"):
            batch_end = min(batch_start + batch_size, n_valid)
            batch_coords = (
                valid_coords[0][batch_start:batch_end],
                valid_coords[1][batch_start:batch_end]
            )
            
            # 提取批次数据
            batch_satellite = satellite_bands[batch_coords]  # (batch_size, 4)
            
            # 计算光谱指数
            batch_indices = []
            for pixel in batch_satellite:
                band_dict = {
                    'G': pixel[0],
                    'R': pixel[1],
                    'REG': pixel[2],
                    'NIR': pixel[3]
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
                batch_indices.append(spectral_indices)
            
            batch_indices = np.array(batch_indices)  # (batch_size, 26)
            
            # 标准化（如果需要）
            if scaler_satellite is not None:
                batch_satellite_norm = scaler_satellite.transform(batch_satellite)
            else:
                batch_satellite_norm = batch_satellite
            
            # 转换为tensor
            satellite_t = torch.FloatTensor(batch_satellite_norm).to(device)
            indices_t = torch.FloatTensor(batch_indices).to(device)
            
            # 传感器独热编码
            sensor_onehot = torch.zeros(len(batch_satellite_norm), 2, dtype=torch.float32).to(device)
            sensor_onehot[:, sensor_id] = 1.0
            
            # 波段掩码
            band_mask_t = torch.FloatTensor([band_mask] * len(batch_satellite_norm)).to(device)
            
            # 预测
            with torch.no_grad():
                uav_pred = model(satellite_t, indices_t, sensor_onehot, band_mask_t)
            
            # 反标准化（如果需要）
            if scaler_uav is not None:
                uav_pred = scaler_uav.inverse_transform(uav_pred.cpu().numpy())
            else:
                uav_pred = uav_pred.cpu().numpy()
            
            # 写入输出数组
            for i, (row, col) in enumerate(zip(batch_coords[0], batch_coords[1])):
                uav_bands[row, col] = uav_pred[i]
        
        # 保存输出影像
        logger.info(f"保存输出影像: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 转换为 (bands, height, width) 格式
        uav_bands_output = np.transpose(uav_bands, (2, 0, 1))  # (4, H, W)
        
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=4,
            dtype=rasterio.float32,
            crs=crs,
            transform=transform,
            nodata=np.nan
        ) as dst:
            for i in range(4):
                dst.write(uav_bands_output[i], i + 1)
                dst.set_band_description(i + 1, f'UAV_{["G", "R", "REG", "NIR"][i]}')
        
        logger.info("预测完成！")


def run_stage1_inference(
    model_path: Path,
    model_config: Dict,
    data_config: Dict,
    s2_raster_path: Optional[Path] = None,
    l8_raster_path: Optional[Path] = None,
    output_dir: Path = Path('outputs/stage1_inference'),
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> None:
    """
    运行Stage 1推理，生成校正后的影像。
    
    参数:
        model_path: 模型权重文件路径
        model_config: 模型配置
        data_config: 数据配置
        s2_raster_path: S2影像路径（可选）
        l8_raster_path: L8影像路径（可选）
        output_dir: 输出目录
        device: 设备
    """
    logger.info("=" * 60)
    logger.info("Stage 1 推理流程")
    logger.info("=" * 60)
    
    # 加载模型
    model = load_stage1_model(model_path, model_config, device)
    
    # 加载标准化器
    scalers_dir = Path('outputs/aligned_images/scalers')
    scaler_s2 = None
    scaler_l8 = None
    scaler_uav = None
    
    if scalers_dir.exists():
        logger.info(f"加载标准化器: {scalers_dir}")
        from data_preprocessing.normalization import StandardScaler
        scaler_s2_obj = StandardScaler()
        scaler_s2_obj.load(scalers_dir / 'scaler_s2.pkl')
        scaler_s2 = scaler_s2_obj
        
        scaler_l8_obj = StandardScaler()
        scaler_l8_obj.load(scalers_dir / 'scaler_l8.pkl')
        scaler_l8 = scaler_l8_obj
        
        scaler_uav_obj = StandardScaler()
        scaler_uav_obj.load(scalers_dir / 'scaler_uav.pkl')
        scaler_uav = scaler_uav_obj
    else:
        logger.warning("未找到标准化器，将使用未标准化的数据")
    
    # 获取波段映射配置
    band_mapping_config = data_config.get('bands', {}).get('satellite', {})
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理S2影像
    if s2_raster_path and s2_raster_path.exists():
        logger.info("\n处理S2影像...")
        output_s2 = output_dir / 's2_corrected_uav_spectrum.tif'
        predict_raster(
            model=model,
            raster_path=s2_raster_path,
            band_mapping_config=band_mapping_config,
            sensor_id=0,
            output_path=output_s2,
            scaler_satellite=scaler_s2,
            scaler_uav=scaler_uav,
            device=device
        )
    
    # 处理L8影像
    if l8_raster_path and l8_raster_path.exists():
        logger.info("\n处理L8影像...")
        output_l8 = output_dir / 'l8_corrected_uav_spectrum.tif'
        predict_raster(
            model=model,
            raster_path=l8_raster_path,
            band_mapping_config=band_mapping_config,
            sensor_id=1,
            output_path=output_l8,
            scaler_satellite=scaler_l8,
            scaler_uav=scaler_uav,
            device=device
        )
    
    logger.info("\n推理完成！")


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

