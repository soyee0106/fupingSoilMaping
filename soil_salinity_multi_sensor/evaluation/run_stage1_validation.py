"""
Stage 1 验证实验主脚本

运行两个验证实验并生成报告。
"""

import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json

from evaluation.stage1_validation import (
    load_stage1_model,
    experiment1_basic_mapping_accuracy,
    experiment2_cross_sensor_consistency
)
from evaluation.stage1_visualization import (
    experiment3_spectral_curves_visualization,
    experiment4_bias_visualization
)
from data_preprocessing.normalization_utils import load_scalers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_validation_experiments(
    model_path: Path,
    model_config_path: Path,
    data_config_path: Path,
    val_data_path: Path,
    output_dir: Path = Path('outputs/stage1_validation')
) -> None:
    """
    运行Stage 1验证实验。
    
    参数:
        model_path: 模型权重文件路径
        model_config_path: 模型配置文件路径
        data_config_path: 数据配置文件路径
        val_data_path: 验证集数据路径
        output_dir: 输出目录
    """
    logger.info("=" * 80)
    logger.info("Stage 1 验证实验")
    logger.info("=" * 80)
    
    # 加载配置
    with open(model_config_path, 'r', encoding='utf-8') as f:
        model_config = yaml.safe_load(f)
    
    with open(data_config_path, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    
    # 加载验证数据
    logger.info(f"加载验证数据: {val_data_path}")
    val_df = pd.read_csv(val_data_path)
    logger.info(f"验证集样本数: {len(val_df)}")
    
    # 加载模型
    device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
    model = load_stage1_model(model_path, model_config, device)
    
    # 加载标准化器
    scalers_dir = Path('outputs/aligned_images/scalers')
    scaler_s2 = None
    scaler_l8 = None
    scaler_uav = None
    
    if scalers_dir.exists():
        logger.info(f"加载标准化器: {scalers_dir}")
        from data_preprocessing.normalization import StandardScaler
        
        if (scalers_dir / 'scaler_s2.pkl').exists():
            scaler_s2_obj = StandardScaler()
            scaler_s2_obj.load(scalers_dir / 'scaler_s2.pkl')
            scaler_s2 = scaler_s2_obj
        
        if (scalers_dir / 'scaler_l8.pkl').exists():
            scaler_l8_obj = StandardScaler()
            scaler_l8_obj.load(scalers_dir / 'scaler_l8.pkl')
            scaler_l8 = scaler_l8_obj
        
        if (scalers_dir / 'scaler_uav.pkl').exists():
            scaler_uav_obj = StandardScaler()
            scaler_uav_obj.load(scalers_dir / 'scaler_uav.pkl')
            scaler_uav = scaler_uav_obj
    else:
        logger.warning("未找到标准化器，将假设数据已标准化")
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"输出目录: {output_dir.absolute()}")
    logger.info(f"输出目录存在: {output_dir.exists()}")
    
    # 实验1：基础映射精度
    logger.info("\n" + "=" * 80)
    logger.info("开始实验1：基础映射精度评估")
    logger.info("=" * 80)
    
    exp1_results = experiment1_basic_mapping_accuracy(
        model=model,
        val_df=val_df,
        scaler_s2=scaler_s2,
        scaler_l8=scaler_l8,
        scaler_uav=scaler_uav,
        device=device
    )
    
    # 实验2：跨传感器一致性
    logger.info("\n" + "=" * 80)
    logger.info("开始实验2：跨传感器一致性评估")
    logger.info("=" * 80)
    
    exp2_results = experiment2_cross_sensor_consistency(
        model=model,
        val_df=val_df,
        scaler_s2=scaler_s2,
        scaler_l8=scaler_l8,
        device=device
    )
    
    # 实验3：光谱曲线可视化
    logger.info("\n" + "=" * 80)
    logger.info("开始实验3：光谱曲线可视化")
    logger.info("=" * 80)
    
    # 获取对齐后的影像路径
    aligned_images_dir = Path('outputs/aligned_images')
    s2_aligned_path = aligned_images_dir / 's2_aligned_normalized.tif'
    l8_aligned_path = aligned_images_dir / 'l8_aligned_normalized.tif'
    uav_aligned_path = aligned_images_dir / 'uav_aligned_normalized.tif'
    
    # 如果标准化影像不存在，尝试使用未标准化的
    if not s2_aligned_path.exists():
        s2_aligned_path = aligned_images_dir / 's2_aligned.tif'
    if not l8_aligned_path.exists():
        l8_aligned_path = aligned_images_dir / 'l8_aligned.tif'
    if not uav_aligned_path.exists():
        uav_aligned_path = aligned_images_dir / 'uav_aligned.tif'
    
    # 获取地类验证点路径
    points_shapefile = Path('originData/地类验证点.shp')
    if not points_shapefile.is_absolute():
        # 相对路径，从项目根目录开始
        project_root = Path(__file__).parent.parent
        points_shapefile = project_root / points_shapefile
    
    if points_shapefile.exists() and s2_aligned_path.exists() and l8_aligned_path.exists() and uav_aligned_path.exists():
        # 获取波段映射配置（支持两种配置结构）
        if 'band_mapping' in data_config:
            band_mapping_config = data_config.get('band_mapping', {})
        elif 'bands' in data_config and 'satellite' in data_config['bands']:
            # 从 bands.satellite 结构获取
            band_mapping_config = {
                's2': data_config['bands']['satellite'].get('s2', {}),
                'l8': data_config['bands']['satellite'].get('l8', {})
            }
        else:
            band_mapping_config = {}
        
        logger.info(f"波段映射配置结构: {list(band_mapping_config.keys())}")
        
        experiment3_spectral_curves_visualization(
            model=model,
            points_shapefile=points_shapefile,
            s2_raster_path=s2_aligned_path,
            l8_raster_path=l8_aligned_path,
            uav_raster_path=uav_aligned_path,
            band_mapping_config=band_mapping_config,
            scaler_s2=scaler_s2,
            scaler_l8=scaler_l8,
            scaler_uav=scaler_uav,
            output_dir=output_dir / 'spectral_curves',
            device=device
        )
        logger.info("实验3完成")
    else:
        logger.warning("跳过实验3：缺少必要的文件")
        logger.warning(f"  地类验证点: {points_shapefile} (存在: {points_shapefile.exists()})")
        logger.warning(f"  S2影像: {s2_aligned_path} (存在: {s2_aligned_path.exists()})")
        logger.warning(f"  L8影像: {l8_aligned_path} (存在: {l8_aligned_path.exists()})")
        logger.warning(f"  UAV影像: {uav_aligned_path} (存在: {uav_aligned_path.exists()})")
    
    # 实验4：偏差可视化
    logger.info("\n" + "=" * 80)
    logger.info("开始实验4：偏差可视化")
    logger.info("=" * 80)
    
    experiment4_bias_visualization(
        model=model,
        val_df=val_df,
        scaler_s2=scaler_s2,
        scaler_l8=scaler_l8,
        output_dir=output_dir / 'bias_analysis',
        device=device,
        n_samples=100
    )
    logger.info("实验4完成")
    
    # 保存结果（转换numpy类型为Python原生类型）
    def convert_to_python_types(obj):
        """递归转换numpy类型为Python原生类型"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_python_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python_types(item) for item in obj]
        else:
            return obj
    
    results = {
        'experiment1_basic_mapping_accuracy': convert_to_python_types(exp1_results),
        'experiment2_cross_sensor_consistency': convert_to_python_types(exp2_results)
    }
    
    results_json_path = output_dir / 'validation_results.json'
    try:
        with open(results_json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"\n结果已保存到: {results_json_path.absolute()}")
    except Exception as e:
        logger.error(f"保存JSON结果失败: {e}")
        logger.error(f"输出目录: {output_dir.absolute()}")
        logger.error(f"输出目录存在: {output_dir.exists()}")
        raise
    
    # 生成文本报告
    report_path = output_dir / 'validation_report.txt'
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Stage 1 验证实验报告\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("实验1：基础映射精度评估\n")
            f.write("-" * 80 + "\n")
            for sensor, metrics in exp1_results.items():
                f.write(f"\n{sensor}:\n")
                f.write(f"  总体R²: {metrics['R2_overall']:.4f}\n")
                f.write(f"  总体RMSE: {metrics['RMSE_overall']:.4f}\n")
                f.write(f"  各波段R²: {metrics['R2_per_band']}\n")
                f.write(f"  各波段RMSE: {metrics['RMSE_per_band']}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("实验2：跨传感器一致性评估\n")
            f.write("-" * 80 + "\n")
            f.write(f"平均欧氏距离: {exp2_results['mean_euclidean_distance']:.4f}\n")
            f.write(f"标准差: {exp2_results['std_euclidean_distance']:.4f}\n")
            f.write(f"中位数: {exp2_results['median_euclidean_distance']:.4f}\n")
            f.write(f"各波段平均绝对差异: {exp2_results['mean_absolute_diff_per_band']}\n")
        
        logger.info(f"报告已保存到: {report_path.absolute()}")
    except Exception as e:
        logger.error(f"保存文本报告失败: {e}")
        logger.error(f"输出目录: {output_dir.absolute()}")
        raise
    
    # 打印摘要
    logger.info("\n" + "=" * 80)
    logger.info("验证实验摘要")
    logger.info("=" * 80)
    logger.info("\n实验1：基础映射精度")
    for sensor, metrics in exp1_results.items():
        logger.info(f"  {sensor}: R²={metrics['R2_overall']:.4f}, RMSE={metrics['RMSE_overall']:.4f}")
    
    logger.info("\n实验2：跨传感器一致性")
    logger.info(f"  平均欧氏距离: {exp2_results['mean_euclidean_distance']:.4f} ± {exp2_results['std_euclidean_distance']:.4f}")
    
    logger.info("\n验证实验完成！")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='运行Stage 1验证实验')
    parser.add_argument('--model_path', type=str, required=True, help='模型权重文件路径')
    parser.add_argument('--model_config', type=str, default='configs/model_config.yaml', help='模型配置文件路径')
    parser.add_argument('--data_config', type=str, default='configs/data_config.yaml', help='数据配置文件路径')
    parser.add_argument('--val_data', type=str, default='outputs/val_data.csv', help='验证集数据路径')
    parser.add_argument('--output_dir', type=str, default='outputs/stage1_validation', help='输出目录')
    
    args = parser.parse_args()
    
    run_validation_experiments(
        model_path=Path(args.model_path),
        model_config_path=Path(args.model_config),
        data_config_path=Path(args.data_config),
        val_data_path=Path(args.val_data),
        output_dir=Path(args.output_dir)
    )

