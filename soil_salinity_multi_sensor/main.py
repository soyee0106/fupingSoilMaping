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

from data_preprocessing import DataPairer, StandardScaler
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
    
    # 读取配置
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
    # 读取数据
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    salinity_column = data_config['column_names']['salinity']
    
    # 提取特征和标签（根据模型类型）
    if model_type == 'stage1':
        # 阶段一：需要卫星特征和真实无人机波段
        # 这里需要根据实际数据列名调整
        # 假设列名格式：S2_B2, S2_B3, ..., S2_NDVI, ..., UAV_B, UAV_G, ...
        pass  # 需要根据实际数据结构实现
    elif model_type == 'stage2':
        # 阶段二：需要无人机波段和盐分值
        pass  # 需要根据实际数据结构实现
    else:
        # 完整模型或基线模型：需要卫星特征和盐分值
        pass  # 需要根据实际数据结构实现
    
    # 这里是一个简化的示例，实际需要根据数据列名进行特征提取
    # 返回占位符
    return None, None, None, None


def train_models(data_config: Dict, model_config: Dict):
    """模型训练模式。"""
    logger.info("=" * 60)
    logger.info("Model Training Mode")
    logger.info("=" * 60)
    
    device = model_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 这里需要根据实际数据格式实现完整的训练流程
    # 由于数据格式可能因项目而异，这里提供一个框架
    
    logger.info("Training pipeline framework (needs data-specific implementation)")
    
    # 示例：训练阶段一
    # stage1_config = model_config['stage1']
    # stage1_model = SensorBiasDecoder(**stage1_config)
    # train_stage1_decoder(...)
    
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


def main():
    """主函数。"""
    parser = argparse.ArgumentParser(
        description='基于光谱解耦的多传感器协同土壤盐分反演系统'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['preprocess', 'train', 'evaluate', 'apply'],
        help='运行模式：preprocess（预处理）、train（训练）、evaluate（评估）、apply（应用）'
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
    
    args = parser.parse_args()
    
    # 加载配置
    data_config = load_config(Path(args.data_config))
    model_config = load_config(Path(args.model_config))
    
    # 根据模式执行相应操作
    if args.mode == 'preprocess':
        preprocess_data(data_config)
    
    elif args.mode == 'train':
        train_models(data_config, model_config)
    
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


if __name__ == '__main__':
    main()

