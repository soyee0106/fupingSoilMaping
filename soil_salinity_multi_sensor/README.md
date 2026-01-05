# 基于光谱解耦的多传感器协同土壤盐分反演

## 项目简介

本项目实现了一个两阶段深度学习模型（Model C），用于多传感器协同的土壤盐分反演。通过光谱解耦网络学习并校正不同卫星（Sentinel-2, Landsat-8）的光谱偏差，将卫星光谱特征转换为模拟的无人机波段反射率，然后使用盐分反演网络预测土壤盐分值。

## 核心特性

- **两阶段深度学习架构**：
  - 阶段一：差异解耦网络（传感器偏差解码器）
  - 阶段二：盐分反演网络
  
- **多传感器支持**：支持Sentinel-2和Landsat-8数据

- **基线模型对比**：包含Model A和Model B作为对比基线

- **完整的训练流程**：预训练、微调、评估一体化

- **模块化设计**：代码高度模块化，便于维护和扩展

## 项目结构

```
soil_salinity_multi_sensor/
├── data_preprocessing/      # 数据预处理模块
│   ├── band_matching.py     # 波段匹配
│   ├── spectral_indices.py   # 光谱指数计算
│   ├── data_pairing.py      # 数据配对（核心）
│   └── normalization.py     # 数据归一化
│
├── models/                   # 模型定义模块
│   ├── stage1_decoder.py    # 阶段一网络
│   ├── stage2_inverter.py   # 阶段二网络
│   ├── full_model.py        # 完整模型（Model C）
│   └── baseline_models.py   # 基线模型（Model A, B）
│
├── training/                 # 训练模块
│   ├── pretrain_stage1.py   # 阶段一预训练
│   ├── pretrain_stage2.py   # 阶段二预训练
│   ├── joint_finetune.py    # 联合微调（核心）
│   └── train_baselines.py   # 基线模型训练
│
├── evaluation/               # 评估与可视化模块
│   ├── metrics.py           # 评估指标
│   ├── uncertainty_analysis.py  # 不确定性分析
│   └── visualization.py     # 可视化
│
├── configs/                 # 配置文件
│   ├── data_config.yaml     # 数据配置
│   └── model_config.yaml    # 模型配置
│
├── main.py                  # 主程序入口
├── requirements.txt         # 项目依赖
└── README.md               # 项目说明
```

## 快速开始

### 1. 环境安装

```bash
# 克隆或下载项目
cd soil_salinity_multi_sensor

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置数据路径

编辑 `configs/data_config.yaml`，设置数据路径：

```yaml
data_paths:
  samples_shapefile: "data/samples.shp"
  s2_raster: "data/sentinel2.tif"
  l8_raster: "data/landsat8.tif"
  uav_raster: "data/uav.tif"
```

### 3. 数据预处理

```bash
python main.py --mode preprocess \
    --data_config configs/data_config.yaml
```

### 4. 模型训练

```bash
# 训练完整模型（Model C）
python main.py --mode train \
    --data_config configs/data_config.yaml \
    --model_config configs/model_config.yaml
```

### 5. 模型评估

```bash
python main.py --mode evaluate \
    --data_config configs/data_config.yaml \
    --model_config configs/model_config.yaml
```

### 6. 大区域应用

```bash
python main.py --mode apply \
    --data_config configs/data_config.yaml \
    --model_config configs/model_config.yaml \
    --raster_path data/region.tif \
    --output_path outputs/prediction.tif
```

## 使用示例

### 数据预处理

```python
from data_preprocessing import DataPairer

pairer = DataPairer()
df = pairer.create_training_data(
    samples_shapefile="data/samples.shp",
    salinity_column="全盐量",
    s2_raster_path="data/sentinel2.tif",
    l8_raster_path="data/landsat8.tif",
    uav_raster_path="data/uav.tif"
)
```

### 模型训练

```python
from models import FullModelC
from training import fine_tune_full_model

# 创建模型
model = FullModelC(
    stage1_config={...},
    stage2_config={...}
)

# 训练
history = fine_tune_full_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    stage1_lr=1e-5,
    stage2_lr=1e-4
)
```

### 模型评估

```python
from evaluation import calculate_all_metrics, plot_scatter

# 计算指标
metrics = calculate_all_metrics(y_true, y_pred)
print(f"R²: {metrics['R2']:.4f}")
print(f"RMSE: {metrics['RMSE']:.4f}")

# 绘制散点图
plot_scatter(y_true, y_pred, metrics, save_path="outputs/scatter.png")
```

## 模型架构

### Model C（完整模型）

1. **阶段一（差异解耦网络）**：
   - 输入：卫星波段 + 光谱指数 + 传感器标签
   - 输出：模拟的无人机4个波段反射率
   - 目的：学习并校正传感器光谱偏差

2. **阶段二（盐分反演网络）**：
   - 输入：模拟无人机波段 + 光谱指数
   - 输出：土壤盐分值
   - 目的：从光谱特征预测盐分

### 基线模型

- **Model A**：直接从卫星特征预测盐分（单阶段）
- **Model B**：卫星→无人机波段→盐分（两阶段，但不学习传感器偏差）

## 训练策略

1. **阶段一预训练**：使用"卫星特征-真实无人机波段"数据对
2. **阶段二预训练**：使用"真实无人机波段-盐分值"数据对
3. **联合微调**：端到端微调，阶段一使用较低学习率（1e-5），阶段二使用较高学习率（1e-4）

## 评估指标

- R²（决定系数）
- RMSE（均方根误差）
- MAE（平均绝对误差）
- MAPE（平均绝对百分比误差）
- 相关系数

## 注意事项

1. **数据格式**：确保输入数据格式符合要求（Shapefile、GeoTIFF等）
2. **坐标系**：确保所有数据使用相同的坐标系
3. **数据质量**：检查数据中的缺失值和异常值
4. **GPU支持**：如果有GPU，建议使用CUDA加速训练

## 许可证

本项目仅供研究使用。

## 联系方式

如有问题或建议，请提交Issue或联系项目维护者。

