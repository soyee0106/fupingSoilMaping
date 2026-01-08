"""
主程序入口

实现命令行接口，可以通过参数控制运行模式：
- preprocess: 数据预处理
- train: 模型训练
- evaluate: 模型评估
- apply: 大区域应用
"""

import argparse
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
import logging
from torch.utils.data import DataLoader, TensorDataset

# 导入项目模块
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data_preprocessing import DataPairer, StandardScaler, process_uav_alignment, process_l8_alignment
from models import (
    SensorBiasDecoder,
    SalinityInverter,
    FullModelC,
    BaselineModelA,
    BaselineModelB
)
from training import (
    train_stage1_decoder,
    train_stage2_inverter,
    fine_tune_full_model,
    train_baseline_a,
    train_baseline_b
)
from evaluation import (
    calculate_all_metrics,
    plot_loss_curves,
    plot_scatter,
    plot_salinity_distribution
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict:
    """加载配置文件。"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def preprocess_data(data_config: Dict):
    """数据预处理模式。"""
    logger.info("=" * 60)
    logger.info("Data Preprocessing Mode")
    logger.info("=" * 60)
    
    # 创建DataPairer
    pairer = DataPairer()
    
    # 检查是否使用密集映射模式
    dense_config = data_config.get('preprocessing', {}).get('dense_mapping', {})
    use_dense_mapping = dense_config.get('enabled', False)
    
    if use_dense_mapping:
        logger.info("Using DENSE MAPPING mode (pixel-to-pixel)")
        logger.info("This will align images and extract all valid pixel pairs")
        
        # 密集映射模式：对齐影像并提取所有像元对
        s2_path = Path(data_config['data_paths']['s2_raster']) if data_config['data_paths'].get('s2_raster') else None
        l8_path = Path(data_config['data_paths']['l8_raster']) if data_config['data_paths'].get('l8_raster') else None
        uav_path = Path(data_config['data_paths']['uav_raster']) if data_config['data_paths'].get('uav_raster') else None
        
        if s2_path is None or uav_path is None:
            raise ValueError("Dense mapping mode requires both S2 and UAV raster paths")
        
        # 读取密集映射配置
        target_resolution = dense_config.get('target_resolution')
        s2_nodata = dense_config.get('s2_nodata', 0)
        l8_nodata = dense_config.get('l8_nodata')
        uav_nodata = dense_config.get('uav_nodata', 65535)
        s2_band_indices = dense_config.get('satellite_band_indices')
        l8_band_indices = dense_config.get('satellite_band_indices')  # L8使用相同的配置
        uav_band_indices = dense_config.get('uav_band_indices')
        
        # 读取波段映射配置（用于确定波段索引）
        band_mapping_config = data_config.get('bands', {}).get('satellite', {})
        sensor_labels = data_config.get('sensor_labels', {'S2': 0, 'L8': 1})
        
        # 如果没有指定波段索引，从配置中获取
        if s2_band_indices is None and 's2' in band_mapping_config:
            s2_mapping = band_mapping_config['s2'].get('band_mapping', {})
            # 根据映射配置确定波段索引（G, R, REG, NIR）
            s2_band_indices = []
            for band_name in ['G', 'R', 'REG', 'NIR']:
                if band_name in s2_mapping:
                    indices = s2_mapping[band_name].get('indices', [])
                    if indices:
                        s2_band_indices.append(indices[0])
        
        if l8_band_indices is None and 'l8' in band_mapping_config:
            l8_mapping = band_mapping_config['l8'].get('band_mapping', {})
            # 根据映射配置确定波段索引（G, R, REG, NIR）
            l8_band_indices = []
            for band_name in ['G', 'R', 'REG', 'NIR']:
                if band_name in l8_mapping:
                    indices = l8_mapping[band_name].get('indices', [])
                    if indices:
                        l8_band_indices.append(indices[0])
        
        if uav_band_indices is None:
            # UAV通常是顺序的4个波段
            uav_band_indices = [1, 2, 3, 4]
        
        # 设置输出路径
        output_dir = Path('outputs/aligned_images')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_l8_tif = output_dir / 'l8_aligned.tif' if l8_path else None
        output_uav_tif = output_dir / 'uav_aligned.tif'
        output_pairs_csv = output_dir / 'pixel_pairs.csv'
        
        # 标准化相关输出路径
        normalize = dense_config.get('normalize', True)
        normalization_method = dense_config.get('normalization_method', 'standard')
        output_scalers_dir = output_dir / 'scalers' if normalize else None
        output_normalized_s2_tif = output_dir / 's2_aligned_normalized.tif' if normalize else None
        output_normalized_l8_tif = output_dir / 'l8_aligned_normalized.tif' if (normalize and l8_path) else None
        output_normalized_uav_tif = output_dir / 'uav_aligned_normalized.tif' if normalize else None
        output_normalized_csv = output_dir / 'pixel_pairs_normalized.csv' if normalize else None
        output_distribution_plots_dir = output_dir / 'distribution_plots' if normalize else None
        
        # 创建密集训练数据
        df = pairer.create_dense_training_data(
            s2_raster_path=s2_path,
            l8_raster_path=l8_path,
            uav_raster_path=uav_path,
            target_resolution=target_resolution,
            s2_band_indices=s2_band_indices,
            l8_band_indices=l8_band_indices,
            uav_band_indices=uav_band_indices,
            s2_nodata=s2_nodata,
            l8_nodata=l8_nodata,
            uav_nodata=uav_nodata,
            band_mapping_config=band_mapping_config,
            output_l8_tif=output_l8_tif,
            output_uav_tif=output_uav_tif,
            output_csv=output_pairs_csv,
            sensor_labels=sensor_labels,
            normalize=normalize,
            normalization_method=normalization_method,
            output_scalers_dir=output_scalers_dir,
            output_normalized_s2_tif=output_normalized_s2_tif,
            output_normalized_l8_tif=output_normalized_l8_tif,
            output_normalized_uav_tif=output_normalized_uav_tif,
            output_normalized_csv=output_normalized_csv,
            output_distribution_plots_dir=output_distribution_plots_dir
        )
        
        # 如果进行了标准化，优先使用标准化后的CSV文件（确保数据一致性）
        if normalize and output_normalized_csv and output_normalized_csv.exists():
            logger.info(f"检测到标准化后的数据文件: {output_normalized_csv}")
            logger.info("使用标准化后的数据文件进行后续处理...")
            df = pd.read_csv(output_normalized_csv)
            logger.info(f"已从标准化CSV加载 {len(df)} 个样本")
        
        # 注意：密集映射模式下没有盐分值，因为这是用于学习光谱映射关系的
        # 如果需要盐分值，需要后续从样点位置提取或使用其他方法
        
    else:
        logger.info("Using SAMPLE POINT mode (extract values at sample locations)")
        
        # 样点提取模式：从样点位置提取值
        samples_path = Path(data_config['data_paths']['samples_shapefile'])
        s2_path = Path(data_config['data_paths']['s2_raster']) if data_config['data_paths'].get('s2_raster') else None
        l8_path = Path(data_config['data_paths']['l8_raster']) if data_config['data_paths'].get('l8_raster') else None
        uav_path = Path(data_config['data_paths']['uav_raster']) if data_config['data_paths'].get('uav_raster') else None
        
        salinity_column = data_config['column_names']['salinity']
        
        # 读取波段映射配置
        band_mapping_config = data_config.get('bands', {}).get('satellite', {})
        
        # 创建训练数据
        df = pairer.create_training_data(
            samples_shapefile=samples_path,
            salinity_column=salinity_column,
            s2_raster_path=s2_path,
            l8_raster_path=l8_path,
            uav_raster_path=uav_path,
            band_mapping_config=band_mapping_config
        )
    
    # 保存预处理数据
    output_path = Path(data_config['output_paths']['preprocessed_data'])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    logger.info(f"Preprocessed data saved to {output_path}")
    
    # 数据划分
    from sklearn.model_selection import train_test_split
    
    train_ratio = data_config['data_split']['train_ratio']
    val_ratio = data_config['data_split']['val_ratio']
    test_ratio = data_config['data_split']['test_ratio']
    random_state = data_config['data_split']['random_state']
    
    # 首先划分训练集和测试集
    train_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        random_state=random_state
    )
    
    # 然后从训练集中划分验证集
    val_size = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        train_df,
        test_size=val_size,
        random_state=random_state
    )
    
    # 保存划分后的数据
    train_path = Path(data_config['output_paths']['train_data'])
    val_path = Path(data_config['output_paths']['val_data'])
    test_path = Path(data_config['output_paths']['test_data'])
    
    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_path, index=False, encoding='utf-8-sig')
    val_df.to_csv(val_path, index=False, encoding='utf-8-sig')
    test_df.to_csv(test_path, index=False, encoding='utf-8-sig')
    
    logger.info(f"Train data: {len(train_df)} samples")
    logger.info(f"Validation data: {len(val_df)} samples")
    logger.info(f"Test data: {len(test_df)} samples")
    
    logger.info("Data preprocessing completed!")


def prepare_data_loaders(
    train_path: Path,
    val_path: Path,
    test_path: Path,
    data_config: Dict,
    model_type: str = 'full'
) -> tuple:
    """
    准备数据加载器。
    
    参数:
        train_path: 训练数据路径
        val_path: 验证数据路径
        test_path: 测试数据路径
        data_config: 数据配置
        model_type: 模型类型（'full', 'stage1', 'stage2', 'baseline_a', 'baseline_b'）
    
    返回:
        (train_loader, val_loader, test_loader, scalers)
    """
    from torch.utils.data import DataLoader
    from data_preprocessing.dataset import DenseMappingDataset, SalinityDataset, FullModelDataset
    
    # 读取数据
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # 检查是否使用密集映射模式（通过检查是否有UAV_band列）
    is_dense_mapping = any('UAV_band' in col for col in train_df.columns)
    
    if model_type == 'stage1':
        # 阶段一：学习从卫星光谱到无人机光谱的映射
        if not is_dense_mapping:
            raise ValueError("Stage 1 training requires dense mapping data (UAV bands)")
        
        # 提取卫星和无人机波段列
        sat_band_cols = [col for col in train_df.columns if col.startswith('SAT_band_')]
        uav_band_cols = [col for col in train_df.columns if col.startswith('UAV_band_')]
        
        sat_band_cols.sort()
        uav_band_cols.sort()
        
        logger.info(f"Satellite bands: {sat_band_cols}")
        logger.info(f"UAV bands: {uav_band_cols}")
        
        # 检查是否有sensor_id列
        sensor_id_col = 'sensor_id' if 'sensor_id' in train_df.columns else None
        if sensor_id_col:
            logger.info(f"Using sensor_id column from data")
        else:
            logger.warning("No sensor_id column found, will use default sensor_id=0")
        
        # 生成掩码向量（根据波段配置，但实际掩码会根据sensor_id动态生成）
        # 这里提供一个默认掩码，但数据集会根据sensor_id自动调整
        band_mask = [1.0, 1.0, 1.0, 1.0]  # 默认S2掩码（G, R, REG, NIR）
        
        # 检查数据是否已经标准化（通过检查数据范围）
        # 如果数据在-5到5之间，认为已经标准化
        sample_sat_values = train_df[sat_band_cols].iloc[0].values
        sample_uav_values = train_df[uav_band_cols].iloc[0].values if uav_band_cols else []
        data_already_normalized = (
            np.all(np.abs(sample_sat_values) < 10) and 
            (len(sample_uav_values) == 0 or np.all(np.abs(sample_uav_values) < 10))
        )
        
        if data_already_normalized:
            logger.info("检测到数据已经标准化，将跳过数据集中的标准化步骤")
        
        # 创建数据集（数据集会自动根据sensor_id生成掩码）
        train_dataset = DenseMappingDataset(
            train_df, sat_band_cols, uav_band_cols, 
            sensor_id_col=sensor_id_col if sensor_id_col else 'sensor_id',
            band_mask=band_mask, 
            normalize=True,
            calculate_indices=True,
            data_already_normalized=data_already_normalized
        )
        val_dataset = DenseMappingDataset(
            val_df, sat_band_cols, uav_band_cols, 
            sensor_id_col=sensor_id_col if sensor_id_col else 'sensor_id',
            band_mask=band_mask, 
            normalize=True, 
            scaler=train_dataset.scaler,
            calculate_indices=True,
            data_already_normalized=data_already_normalized
        )
        test_dataset = DenseMappingDataset(
            test_df, sat_band_cols, uav_band_cols, 
            sensor_id_col=sensor_id_col if sensor_id_col else 'sensor_id',
            band_mask=band_mask,
            normalize=True, 
            scaler=train_dataset.scaler,
            calculate_indices=True,
            data_already_normalized=data_already_normalized
        )
        
        scalers = {'satellite': train_dataset.scaler}
        
    elif model_type == 'stage2':
        # 阶段二：从无人机光谱预测盐分值
        salinity_col = data_config['column_names']['salinity']
        
        # 提取无人机波段和光谱指数
        uav_band_cols = [col for col in train_df.columns if col.startswith('UAV_band_')]
        spectral_index_cols = [col for col in train_df.columns if col.startswith('UAV_') and 'band' not in col]
        
        uav_band_cols.sort()
        spectral_index_cols.sort()
        
        logger.info(f"UAV bands: {uav_band_cols}")
        logger.info(f"Spectral indices: {len(spectral_index_cols)} indices")
        
        # 创建数据集
        train_dataset = SalinityDataset(
            train_df, uav_band_cols, spectral_index_cols, salinity_col, normalize=True
        )
        val_dataset = SalinityDataset(
            val_df, uav_band_cols, spectral_index_cols, salinity_col,
            normalize=True, scaler=train_dataset.scaler
        )
        test_dataset = SalinityDataset(
            test_df, uav_band_cols, spectral_index_cols, salinity_col,
            normalize=True, scaler=train_dataset.scaler
        )
        
        scalers = {'features': train_dataset.scaler}
        
    else:
        # 完整模型或基线模型：从卫星光谱预测盐分值
        salinity_col = data_config['column_names']['salinity']
        
        # 提取卫星波段
        sat_band_cols = [col for col in train_df.columns if col.startswith('SAT_band_')]
        sat_band_cols.sort()
        
        logger.info(f"Satellite bands: {sat_band_cols}")
        
        # 确定传感器ID和掩码
        sensor_labels = data_config.get('sensor_labels', {'S2': 0, 'L8': 1})
        sensor_id = sensor_labels.get('S2', 0)
        
        band_config = data_config.get('bands', {}).get('satellite', {}).get('s2', {})
        band_mapping = band_config.get('band_mapping', {})
        band_mask = [
            1 if 'G' in band_mapping else 0,
            1 if 'R' in band_mapping else 0,
            1 if 'REG' in band_mapping and band_mapping.get('REG', {}).get('indices') else 0,
            1 if 'NIR' in band_mapping else 0
        ]
        
        # 创建数据集
        train_dataset = FullModelDataset(
            train_df, sat_band_cols, sensor_id, band_mask, salinity_col, normalize=True
        )
        val_dataset = FullModelDataset(
            val_df, sat_band_cols, sensor_id, band_mask, salinity_col,
            normalize=True, scaler=train_dataset.scaler
        )
        test_dataset = FullModelDataset(
            test_df, sat_band_cols, sensor_id, band_mask, salinity_col,
            normalize=True, scaler=train_dataset.scaler
        )
        
        scalers = {'satellite': train_dataset.scaler}
    
    # 创建DataLoader
    batch_size = data_config.get('preprocessing', {}).get('batch_size', 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"Data loaders created: batch_size={batch_size}")
    
    return train_loader, val_loader, test_loader, scalers


def train_models(data_config: Dict, model_config: Dict, stage: str = 'stage1'):
    """
    模型训练模式。
    
    参数:
        data_config: 数据配置
        model_config: 模型配置
        stage: 训练阶段 ('stage1', 'stage2', 'full', 'baseline_a', 'baseline_b')
    """
    logger.info("=" * 60)
    logger.info(f"Model Training Mode - {stage.upper()}")
    logger.info("=" * 60)
    
    import torch
    
    device = model_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 准备数据加载器
    train_path = Path(data_config['output_paths']['train_data'])
    val_path = Path(data_config['output_paths']['val_data'])
    test_path = Path(data_config['output_paths']['test_data'])
    
    train_loader, val_loader, test_loader, scalers = prepare_data_loaders(
        train_path, val_path, test_path, data_config, model_type=stage
    )
    
    # 创建输出目录
    output_dir = Path('outputs/models')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if stage == 'stage1':
        # 训练阶段一：传感器偏差解码器
        logger.info("Training Stage 1: Sensor Bias Decoder")
        
        stage1_config = model_config['stage1']
        stage1_model = SensorBiasDecoder(**stage1_config)
        stage1_model = stage1_model.to(device)
        
        training_config = model_config.get('training', {})
        # 使用stage1_pretrain配置
        stage1_training = training_config.get('stage1_pretrain', {})
        
        history = train_stage1_decoder(
            model=stage1_model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=stage1_training.get('num_epochs', 100),
            learning_rate=stage1_training.get('learning_rate', 1e-3),
            device=device,
            save_dir=output_dir / 'stage1',
            save_best=True,
            patience=stage1_training.get('patience', 10)
        )
        
        logger.info(f"Stage 1 training completed!")
        logger.info(f"Best validation loss: {history.get('best_val_loss', 'N/A')}")
        
    elif stage == 'stage2':
        # ========================================================================
        # 【已封存】训练阶段二：盐分反演网络（深度学习模型）
        # 
        # 注意：此功能已封存，不再作为主要实验流程。
        # 主要实验流程请使用：python main.py --mode run_stage2_experiments
        # 该方法使用传统机器学习方法（PLSR、SVR等）进行盐分反演对比实验。
        # ========================================================================
        logger.warning("=" * 60)
        logger.warning("⚠️  警告：Stage 2深度学习训练已封存")
        logger.warning("=" * 60)
        logger.warning("此功能已不再作为主要实验流程。")
        logger.warning("主要实验流程请使用：")
        logger.warning("  python main.py --mode run_stage2_experiments")
        logger.warning("该方法使用传统机器学习方法进行5组对比实验。")
        logger.warning("=" * 60)
        logger.warning("如果确实需要训练深度学习模型，请取消注释下方代码。")
        logger.warning("=" * 60)
        raise NotImplementedError(
            "Stage 2深度学习训练已封存。"
            "请使用 'python main.py --mode run_stage2_experiments' 进行主要实验。"
        )
        
        # # 以下代码已封存，如需使用请取消注释
        # stage2_config = model_config['stage2']
        # stage2_model = SalinityInverter(**stage2_config)
        # stage2_model = stage2_model.to(device)
        # 
        # training_config = model_config.get('training', {})
        # stage2_training = training_config.get('stage2', {})
        # 
        # from training.pretrain_stage2 import train_stage2_inverter
        # 
        # history = train_stage2_inverter(
        #     model=stage2_model,
        #     train_loader=train_loader,
        #     val_loader=val_loader,
        #     num_epochs=stage2_training.get('num_epochs', 100),
        #     learning_rate=stage2_training.get('learning_rate', 1e-3),
        #     device=device,
        #     save_dir=output_dir / 'stage2',
        #     save_best=True,
        #     patience=stage2_training.get('patience', 10)
        # )
        # 
        # logger.info(f"Stage 2 training completed!")
        # logger.info(f"Best validation loss: {history.get('best_val_loss', 'N/A')}")
        
    elif stage == 'full':
        # 联合微调完整模型
        logger.info("Fine-tuning Full Model")
        
        # 加载预训练的阶段一和阶段二模型
        stage1_path = output_dir / 'stage1' / 'best_model.pth'
        stage2_path = output_dir / 'stage2' / 'best_model.pth'
        
        full_config = model_config['full_model']
        full_model = FullModelC(**full_config)
        
        if stage1_path.exists():
            logger.info(f"Loading Stage 1 from {stage1_path}")
            full_model.stage1.load_state_dict(torch.load(stage1_path, map_location=device))
        if stage2_path.exists():
            logger.info(f"Loading Stage 2 from {stage2_path}")
            full_model.stage2.load_state_dict(torch.load(stage2_path, map_location=device))
        
        full_model = full_model.to(device)
        
        training_config = model_config.get('training', {})
        full_training = training_config.get('full_model', {})
        
        from training.joint_finetune import fine_tune_full_model
        
        history = fine_tune_full_model(
            model=full_model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=full_training.get('num_epochs', 50),
            learning_rate=full_training.get('learning_rate', 1e-4),
            device=device,
            save_dir=output_dir / 'full_model',
            save_best=True,
            patience=full_training.get('patience', 10)
        )
        
        logger.info(f"Full model fine-tuning completed!")
        logger.info(f"Best validation loss: {history.get('best_val_loss', 'N/A')}")
        
    else:
        logger.warning(f"Unknown stage: {stage}")
    
    logger.info("Model training completed!")


def evaluate_models(data_config: Dict, model_config: Dict):
    """模型评估模式。"""
    logger.info("=" * 60)
    logger.info("Model Evaluation Mode")
    logger.info("=" * 60)
    
    # 加载测试数据
    test_path = Path(data_config['output_paths']['test_data'])
    test_df = pd.read_csv(test_path)
    
    # 这里需要根据实际模型和数据进行评估
    logger.info("Evaluation pipeline framework (needs model-specific implementation)")
    
    logger.info("Model evaluation completed!")


def apply_to_region(data_config: Dict, model_config: Dict, raster_path: Path, output_path: Path):
    """大区域应用模式。"""
    logger.info("=" * 60)
    logger.info("Regional Application Mode")
    logger.info("=" * 60)
    
    logger.info(f"Input raster: {raster_path}")
    logger.info(f"Output path: {output_path}")
    
    # 这里需要实现大区域预测的流程
    # 1. 读取栅格影像
    # 2. 提取特征
    # 3. 使用模型预测
    # 4. 保存结果
    
    logger.info("Regional application completed!")


def align_images(data_config: Dict, sensor_type: str = 'uav'):
    """
    影像对齐模式：将UAV或L8影像对齐到Sentinel-2。
    
    参数:
        data_config: 数据配置
        sensor_type: 传感器类型，'uav' 或 'l8'
    """
    logger.info("=" * 60)
    logger.info(f"Image Alignment Mode - {sensor_type.upper()}")
    logger.info("=" * 60)
    
    # 获取路径配置
    s2_raster_path = Path(data_config['data_paths']['s2_raster'])
    
    # 创建输出目录
    output_dir = Path('outputs/aligned_images')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if sensor_type == 'uav':
        uav_raster_path = Path(data_config['data_paths']['uav_raster'])
        
        output_tif_path = output_dir / 'uav_aligned_to_s2.tif'
        output_csv_path = output_dir / 'uav_s2_pixel_pairs.csv'
        
        # 获取nodata配置
        preprocessing = data_config.get('preprocessing', {})
        dense_mapping = preprocessing.get('dense_mapping', {})
        uav_nodata = dense_mapping.get('uav_nodata', 65535)
        s2_nodata = dense_mapping.get('s2_nodata', 0)
        
        # 获取目标分辨率（从S2影像读取）
        import rasterio
        with rasterio.open(s2_raster_path) as s2_src:
            target_resolution = abs(s2_src.transform[0])
        
        logger.info(f"UAV影像路径: {uav_raster_path}")
        logger.info(f"Sentinel-2影像路径: {s2_raster_path}")
        logger.info(f"目标分辨率: {target_resolution}")
        logger.info(f"输出TIF: {output_tif_path}")
        logger.info(f"输出CSV: {output_csv_path}")
        
        process_uav_alignment(
            uav_raster_path=uav_raster_path,
            s2_raster_path=s2_raster_path,
            output_tif_path=output_tif_path,
            output_csv_path=output_csv_path,
            uav_nodata=uav_nodata,
            s2_nodata=s2_nodata,
            target_resolution=target_resolution,
        )
        
    elif sensor_type == 'l8':
        l8_raster_path = Path(data_config['data_paths']['l8_raster'])
        
        output_tif_path = output_dir / 'l8_aligned_to_s2.tif'
        output_csv_path = output_dir / 'l8_s2_pixel_pairs.csv'
        
        # 获取nodata配置
        preprocessing = data_config.get('preprocessing', {})
        dense_mapping = preprocessing.get('dense_mapping', {})
        l8_nodata = dense_mapping.get('l8_nodata', None)
        s2_nodata = dense_mapping.get('s2_nodata', 0)
        
        logger.info(f"Landsat-8影像路径: {l8_raster_path}")
        logger.info(f"Sentinel-2影像路径: {s2_raster_path}")
        logger.info(f"输出TIF: {output_tif_path}")
        logger.info(f"输出CSV: {output_csv_path}")
        
        process_l8_alignment(
            l8_raster_path=l8_raster_path,
            s2_raster_path=s2_raster_path,
            output_tif_path=output_tif_path,
            output_csv_path=output_csv_path,
            l8_nodata=l8_nodata,
            s2_nodata=s2_nodata,
        )
    else:
        logger.error(f"未知的传感器类型: {sensor_type}")
        return
    
    logger.info("影像对齐处理完成！")


def main():
    """主函数。"""
    parser = argparse.ArgumentParser(
        description='基于光谱解耦的多传感器协同土壤盐分反演系统'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['preprocess', 'train', 'evaluate', 'apply', 'align', 'infer_stage1', 'validate_stage1', 'run_stage2_experiments'],
        help='运行模式：preprocess（预处理）、train（训练）、evaluate（评估）、apply（应用）、align（影像对齐）、infer_stage1（Stage1推理）、validate_stage1（Stage1验证）、run_stage2_experiments（Stage2对比实验）'
    )
    
    parser.add_argument(
        '--data_config',
        type=str,
        default='configs/data_config.yaml',
        help='数据配置文件路径'
    )
    
    parser.add_argument(
        '--model_config',
        type=str,
        default='configs/model_config.yaml',
        help='模型配置文件路径'
    )
    
    parser.add_argument(
        '--raster_path',
        type=str,
        default=None,
        help='大区域应用时的输入栅格路径'
    )
    
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='输出路径'
    )
    
    parser.add_argument(
        '--stage',
        type=str,
        default='stage1',
        choices=['stage1', 'stage2', 'full', 'baseline_a', 'baseline_b'],
        help='训练阶段：stage1（阶段一）、stage2（阶段二，已封存）、full（联合微调）、baseline_a、baseline_b'
    )
    
    args = parser.parse_args()
    
    # 加载配置
    data_config = load_config(Path(args.data_config))
    model_config = load_config(Path(args.model_config))
    
    # 根据模式执行相应操作
    if args.mode == 'preprocess':
        preprocess_data(data_config)
    
    elif args.mode == 'train':
        train_models(data_config, model_config, stage=args.stage)
    
    elif args.mode == 'evaluate':
        evaluate_models(data_config, model_config)
    
    elif args.mode == 'apply':
        if args.raster_path is None or args.output_path is None:
            logger.error("大区域应用模式需要提供 --raster_path 和 --output_path 参数")
            return
        apply_to_region(
            data_config,
            model_config,
            Path(args.raster_path),
            Path(args.output_path)
        )
    
    elif args.mode == 'align':
        # 影像对齐模式，需要指定传感器类型
        sensor_type = args.stage if args.stage in ['uav', 'l8'] else 'uav'
        if args.stage not in ['uav', 'l8']:
            logger.warning(f"对齐模式默认使用 'uav'，如需处理L8请使用 --stage l8")
        align_images(data_config, sensor_type=sensor_type)
    
    elif args.mode == 'infer_stage1':
        # Stage 1推理模式：生成校正后的影像
        from inference.stage1_inference import run_stage1_inference
        
        model_path = Path('outputs/models/stage1/best_model.pth')
        if not model_path.exists():
            logger.error(f"模型文件不存在: {model_path}")
            logger.error("请先训练Stage 1模型")
            return
        
        s2_raster_path = Path(data_config['data_paths']['s2_raster']) if data_config['data_paths'].get('s2_raster') else None
        l8_raster_path = Path(data_config['data_paths']['l8_raster']) if data_config['data_paths'].get('l8_raster') else None
        
        run_stage1_inference(
            model_path=model_path,
            model_config=model_config,
            data_config=data_config,
            s2_raster_path=s2_raster_path,
            l8_raster_path=l8_raster_path,
            output_dir=Path('outputs/stage1_inference')
        )
    
    elif args.mode == 'validate_stage1':
        from evaluation.run_stage1_validation import run_validation_experiments
        run_validation_experiments(
            data_config_path=args.data_config,
            model_config_path=args.model_config
        )
    
    elif args.mode == 'run_stage2_experiments':
        from evaluation.stage2_experiments import main as run_stage2_experiments
        run_stage2_experiments()


if __name__ == '__main__':
    main()

