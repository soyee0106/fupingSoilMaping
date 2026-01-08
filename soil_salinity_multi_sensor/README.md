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

# 安装依赖（推荐使用虚拟环境）
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 如果使用GPU，确保已安装对应版本的PyTorch
# CUDA 11.8: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# CPU版本: pip install torch torchvision torchaudio
```

### 2. 配置数据路径

编辑 `configs/data_config.yaml`，设置数据路径：

```yaml
data_paths:
  samples_shapefile: "data/samples.shp"      # 样点Shapefile路径（必须包含盐分列）
  s2_raster: "data/sentinel2.tif"           # Sentinel-2影像路径
  l8_raster: "data/landsat8.tif"             # Landsat-8影像路径（可选）
  uav_raster: "data/uav.tif"                # 无人机影像路径

column_names:
  salinity: "全盐量"                         # 盐分列名（根据实际数据修改）

# 数据划分比例
data_split:
  train_ratio: 0.7                          # 训练集比例
  val_ratio: 0.15                           # 验证集比例
  test_ratio: 0.15                          # 测试集比例
  random_state: 42                          # 随机种子
```

编辑 `configs/model_config.yaml`，配置模型参数：

```yaml
device: "cuda"                              # 训练设备：'cuda' 或 'cpu'
stage1:
  n_satellite_bands: 4                      # 卫星波段数
  n_spectral_indices: 26                    # 光谱指数数量
  n_sensors: 2                              # 传感器数量（S2和L8）
  n_uav_bands: 4                            # 无人机波段数
  # ... 更多参数见配置文件

training:
  stage1_pretrain:
    num_epochs: 100                         # 训练轮数
    learning_rate: 0.001                    # 学习率
    patience: 10                             # 早停耐心值
```

### 3. 数据预处理

数据预处理支持两种模式：样点提取模式和密集映射模式。

**样点提取模式**（从样点位置提取光谱值）：
```bash
python main.py --mode preprocess --data_config configs/data_config.yaml
```

**密集映射模式**（像素级对齐，生成密集训练数据）：
需要在 `configs/data_config.yaml` 中启用 `preprocessing.dense_mapping.enabled: true`，然后运行：
```bash
python main.py --mode preprocess --data_config configs/data_config.yaml
```

**影像对齐模式**（单独对齐UAV或L8影像到S2）：
```bash
# 对齐UAV影像到S2
python main.py --mode align --data_config configs/data_config.yaml --stage uav

# 对齐L8影像到S2
python main.py --mode align --data_config configs/data_config.yaml --stage l8
```

### 4. 模型训练

训练支持多个阶段和模型类型：

**阶段一预训练**（传感器偏差解码器）：
```bash
python main.py --mode train --stage stage1 --data_config configs/data_config.yaml --model_config configs/model_config.yaml
```

**阶段二预训练**（已封存，不推荐使用）：
```bash
# ⚠️ 已封存：此功能已不再作为主要实验流程
# python main.py --mode train --stage stage2 --data_config configs/data_config.yaml --model_config configs/model_config.yaml
```

**说明**：
- ⚠️ **此功能已封存**，不再作为主要实验流程
- 主要实验流程请使用 `run_stage2_experiments`（见下方）
- 如需使用深度学习模型，请参考完整模型（Model C）的联合微调流程

**完整模型联合微调**（Model C，端到端训练，可选）：
```bash
python main.py --mode train --stage full --data_config configs/data_config.yaml --model_config configs/model_config.yaml
```

**说明**：
- ⚠️ **此功能需要Stage 2预训练模型，但Stage 2训练已封存，因此此功能目前无法正常使用**
- 联合微调是将预训练好的Stage 1和Stage 2连接，使用"卫星特征→盐分值"数据端到端训练
- Stage 1使用较低学习率（1e-5），Stage 2使用较高学习率（1e-4）
- **与 `run_stage2_experiments` 的关系**：两者是**完全独立**的路径
  - `run_stage2_experiments`：使用传统机器学习方法（PLSR、SVR等），**不需要**联合微调
  - 联合微调：用于训练深度学习完整模型，与 `run_stage2_experiments` 无关

**基线模型训练**（Model A或Model B）：
```bash
# 训练Model A（单阶段基线）
python main.py --mode train --stage baseline_a --data_config configs/data_config.yaml --model_config configs/model_config.yaml

# 训练Model B（两阶段基线，不学习传感器偏差）
python main.py --mode train --stage baseline_b --data_config configs/data_config.yaml --model_config configs/model_config.yaml
```

### 5. 模型评估与验证

**Stage 1验证**（验证光谱校正效果）：
```bash
python main.py --mode validate_stage1 --data_config configs/data_config.yaml --model_config configs/model_config.yaml
```

**Stage 2盐分反演对比实验**（⭐ 主要实验流程）：
```bash
python main.py --mode run_stage2_experiments --data_config configs/data_config.yaml --model_config configs/model_config.yaml
```

**说明**：
- ⭐ **这是主要的盐分反演实验流程**，使用传统机器学习方法进行对比实验
- 运行5个对比实验，每个实验使用多种模型（SVR、RandomForest、PLSR、GradientBoosting、XGBoost）
- 每个实验会生成独立的 `experiment_results.xlsx` 文件，包含详细的模型评估指标
- **实验设计**：
  - 实验1：原始UAV影像反演（30个FP样点）
  - 实验2：原始S2影像反演（108个非FP样点）
  - 实验3：原始L8影像反演（108个非FP样点）
  - 实验4：S2校正后UAV光谱反演（108个非FP样点，需要先运行 `infer_stage1`）
  - 实验5：L8校正后UAV光谱反演（108个非FP样点，需要先运行 `infer_stage1`）
- **输出位置**：
  - 各实验结果：`outputs/stage2_experiments/exp{1-5}_*/experiment_results.xlsx`
  - 对比报告：`outputs/stage2_experiments/experiments_comparison.xlsx`
  - 散点图：`outputs/stage2_experiments/exp{1-5}_*/scatter_*.png`

**Stage 1推理**（生成校正后的卫星影像，用于Stage 2实验）：
```bash
python main.py --mode infer_stage1 --data_config configs/data_config.yaml --model_config configs/model_config.yaml
```

**重要说明**：
- 此步骤会生成 `outputs/stage1_inference/s2_corrected_uav_spectrum.tif` 和 `outputs/stage1_inference/l8_corrected_uav_spectrum.tif`
- 这两个文件是Stage 2对比实验（exp4和exp5）的必需输入
- 必须在运行 `run_stage2_experiments` 之前执行此步骤
- 需要确保 `configs/data_config.yaml` 中配置了 `s2_raster` 和 `l8_raster` 路径

**通用模型评估**：
```bash
python main.py --mode evaluate --data_config configs/data_config.yaml --model_config configs/model_config.yaml
```

### 6. 大区域应用

将训练好的模型应用到整个区域进行盐分预测：
```bash
python main.py --mode apply --data_config configs/data_config.yaml --model_config configs/model_config.yaml --raster_path data/region.tif --output_path outputs/prediction.tif
```

**参数说明**：
- `--raster_path`: 输入的卫星影像路径（Sentinel-2或Landsat-8）
- `--output_path`: 输出的盐分预测栅格路径

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

2. **阶段二（盐分反演网络）**：⚠️ **已封存**
   - 输入：模拟无人机波段 + 光谱指数
   - 输出：土壤盐分值
   - 目的：从光谱特征预测盐分
   - **注意**：此阶段训练已封存，主要实验流程使用传统机器学习方法（见 `run_stage2_experiments`）

### 基线模型

- **Model A**：直接从卫星特征预测盐分（单阶段）
- **Model B**：卫星→无人机波段→盐分（两阶段，但不学习传感器偏差）

## 训练策略

1. **阶段一预训练**：使用"卫星特征-真实无人机波段"数据对
2. **Stage 2对比实验**：使用传统机器学习方法（PLSR、SVR等）进行5组对比实验，评估不同数据源对盐分反演效果的影响
3. **联合微调**（可选）：端到端微调完整模型，阶段一使用较低学习率（1e-5），阶段二使用较高学习率（1e-4）

## 评估指标

- R²（决定系数）
- RMSE（均方根误差）
- MAE（平均绝对误差）
- MAPE（平均绝对百分比误差）
- 相关系数

## 完整命令参考

### 所有可用模式

| 模式 | 说明 | 必需参数 | 可选参数 |
|------|------|----------|----------|
| `preprocess` | 数据预处理 | `--data_config` | - |
| `train` | 模型训练 | `--data_config`, `--model_config` | `--stage` (stage1/stage2已封存/full/baseline_a/baseline_b) |
| `evaluate` | 模型评估 | `--data_config`, `--model_config` | - |
| `apply` | 大区域应用 | `--data_config`, `--model_config`, `--raster_path`, `--output_path` | - |
| `align` | 影像对齐 | `--data_config` | `--stage` (uav/l8) |
| `infer_stage1` | Stage1推理 | `--data_config`, `--model_config` | - |
| `validate_stage1` | Stage1验证 | `--data_config`, `--model_config` | - |
| `run_stage2_experiments` | ⭐ Stage2盐分反演对比实验（主要实验流程） | `--data_config`, `--model_config` | - |

### 常用命令组合示例

**完整训练流程**（从数据预处理到模型训练）：
```bash
# 步骤1: 数据预处理
python main.py --mode preprocess --data_config configs/data_config.yaml

# 步骤2: 训练阶段一（传感器偏差解码器）
python main.py --mode train --stage stage1 --data_config configs/data_config.yaml --model_config configs/model_config.yaml

# 步骤3: Stage 1推理（生成校正后的影像，用于后续实验）
python main.py --mode infer_stage1 --data_config configs/data_config.yaml --model_config configs/model_config.yaml

# 步骤4: 验证Stage1效果（可选，用于评估光谱校正质量）
python main.py --mode validate_stage1 --data_config configs/data_config.yaml --model_config configs/model_config.yaml

# 步骤5: 运行Stage2盐分反演对比实验 ⭐ 主要实验流程
python main.py --mode run_stage2_experiments --data_config configs/data_config.yaml --model_config configs/model_config.yaml
```

**流程说明**：
- **步骤1（数据预处理）**：从样点位置提取光谱值和盐分值，生成训练/验证/测试集
- **步骤2（训练Stage 1）**：训练传感器偏差解码器，学习从卫星光谱到UAV光谱的映射
- **步骤3（Stage 1推理）**：⭐ **必需步骤**，生成校正后的影像文件（`s2_corrected_uav_spectrum.tif` 和 `l8_corrected_uav_spectrum.tif`），这些文件是步骤5中exp4和exp5的必需输入
- **步骤4（验证Stage1）**：可选步骤，用于评估光谱校正效果，生成光谱曲线和偏差分析图
- **步骤5（Stage 2对比实验）**：⭐ **主要实验流程**，使用传统机器学习方法进行5组对比实验，评估不同数据源对盐分反演效果的影响，每个实验生成独立的 `experiment_results.xlsx` 文件
- **重要**：
  - 如果跳过步骤3，步骤5中的exp4（S2校正）和exp5（L8校正）实验将无法运行
  - **联合微调与步骤5无关**：`run_stage2_experiments` 使用传统机器学习方法，不依赖深度学习模型，因此不需要联合微调

**使用自定义配置文件**：
```bash
python main.py --mode train --stage stage1 --data_config my_data_config.yaml --model_config my_model_config.yaml
```

**Windows系统命令**（使用反斜杠）：
```bash
python main.py --mode preprocess --data_config configs\data_config.yaml
```

## 输出文件说明

### 数据预处理输出
- `outputs/preprocessed_data.csv`: 预处理后的完整数据
- `outputs/train_data.csv`: 训练集
- `outputs/val_data.csv`: 验证集
- `outputs/test_data.csv`: 测试集
- `outputs/aligned_images/`: 对齐后的影像（密集映射模式）

### 模型训练输出
- `outputs/models/stage1/best_model.pth`: Stage1最佳模型（传感器偏差解码器）
- `outputs/models/full_model/best_model.pth`: 完整模型最佳权重（可选，用于端到端训练）
- ⚠️ `outputs/models/stage2/best_model.pth`: Stage2深度学习模型（已封存，不再使用）

### Stage 1推理输出（重要）
- `outputs/stage1_inference/s2_corrected_uav_spectrum.tif`: S2校正后的UAV光谱影像（4波段）
- `outputs/stage1_inference/l8_corrected_uav_spectrum.tif`: L8校正后的UAV光谱影像（4波段）
- **注意**：这两个文件是Stage 2对比实验（exp4和exp5）的必需输入，必须通过 `--mode infer_stage1` 生成

### 验证和评估输出
- `outputs/stage1_validation/`: Stage1验证结果（光谱曲线、偏差分析等）
- `outputs/stage2_experiments/`: Stage2对比实验结果
  - `exp1_uav_original/experiment_results.xlsx`: 实验1结果（原始UAV）
  - `exp2_s2_original/experiment_results.xlsx`: 实验2结果（原始S2）
  - `exp3_l8_original/experiment_results.xlsx`: 实验3结果（原始L8）
  - `exp4_s2_corrected/experiment_results.xlsx`: 实验4结果（S2校正后）
  - `exp5_l8_corrected/experiment_results.xlsx`: 实验5结果（L8校正后）
  - `experiments_comparison.xlsx`: 所有实验的对比报告

## 注意事项

1. **数据格式**：
   - Shapefile必须包含几何信息和盐分列
   - 栅格数据必须是GeoTIFF格式，包含地理坐标信息
   - 确保所有数据使用相同的坐标系（建议使用WGS84或UTM）

2. **数据质量**：
   - 检查数据中的缺失值和异常值
   - 确保样点位置在影像覆盖范围内
   - 检查nodata值的设置是否正确

3. **GPU支持**：
   - 如果有GPU，建议使用CUDA加速训练
   - 在`model_config.yaml`中设置`device: "cuda"`
   - 确保PyTorch版本与CUDA版本匹配

4. **内存管理**：
   - 密集映射模式会生成大量数据，注意磁盘空间
   - 大影像处理时注意内存使用情况
   - 可以分批处理或使用数据加载器的batch_size参数

5. **训练建议**：
   - 先运行小规模数据测试流程
   - 根据验证集表现调整超参数
   - 使用早停机制避免过拟合
   - 保存中间检查点以便恢复训练

## 许可证

本项目仅供研究使用。

## 联系方式

如有问题或建议，请提交Issue或联系项目维护者。

