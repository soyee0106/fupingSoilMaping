# Stage 1 训练指南

## 概述

Stage 1（光谱解耦网络）的目标是学习从**卫星光谱**到**无人机光谱**的映射关系，校正不同传感器（Sentinel-2、Landsat-8）的光谱偏差。

## 网络架构

- **输入**：
  - 归一化卫星波段（4个：G, R, REG, NIR）
  - 光谱指数（26个）
  - 传感器ID（独热编码：S2=[1,0], L8=[0,1]）
  - 波段掩码向量（标记有效波段：S2=[1,1,1,1], L8=[1,1,0,1]）

- **输出**：
  - 预测的无人机4个波段反射率（G, R, REG, NIR）

- **网络结构**：
  - 3-4个全连接层（可配置）
  - 使用BatchNorm和Dropout防止过拟合
  - 输出层不使用激活函数（反射率值域可能>1）

## 数据要求

训练数据需要包含：
- `SAT_band_1` 到 `SAT_band_4`：卫星波段（G, R, REG, NIR）
- `UAV_band_1` 到 `UAV_band_4`：无人机波段（G, R, REG, NIR）
- `sensor_id`：传感器ID（0=S2, 1=L8）

数据应该已经通过 `preprocess` 模式生成，保存在：
- `outputs/train_data.csv`
- `outputs/val_data.csv`
- `outputs/test_data.csv`

## 训练命令

```bash
python main.py --mode train \
    --data_config configs/data_config.yaml \
    --model_config configs/model_config.yaml \
    --stage stage1
```

## 训练过程

训练过程会：
1. **自动计算光谱指数**：从卫星波段计算26个光谱指数
2. **自动生成掩码**：根据`sensor_id`自动生成波段掩码（S2全1，L8的REG为0）
3. **数据归一化**：使用StandardScaler对卫星波段进行Z-score标准化
4. **监控指标**：
   - **MSE损失**：衡量预测光谱与真实光谱的差距
   - **R²分数**：决定系数，越接近1越好（更直观）

## 输出结果

训练完成后，模型会保存在：
- `outputs/models/stage1/best_model.pth`：最佳模型（基于验证损失）
- `outputs/models/stage1/stage1_final_model.pth`：最终模型

训练历史包括：
- `train_losses`：训练损失列表
- `val_losses`：验证损失列表
- `train_r2_scores`：训练集R²分数列表
- `val_r2_scores`：验证集R²分数列表

## 配置参数

在 `configs/model_config.yaml` 中可以调整：

```yaml
stage1:
  n_satellite_bands: 4
  n_spectral_indices: 26
  n_sensors: 2
  n_mask_bits: 4
  hidden_dims: [128, 64, 32]  # 隐藏层维度
  dropout_rate: 0.3
  use_batch_norm: true
  use_mask: true

training:
  stage1_pretrain:
    num_epochs: 100
    learning_rate: 0.001
    batch_size: 32
    patience: 10  # 早停耐心值
```

## 训练技巧

1. **学习率**：默认0.001，如果损失不下降可以尝试降低（如0.0001）
2. **早停**：如果验证损失连续10个epoch不下降，训练会自动停止
3. **批次大小**：根据GPU内存调整，默认32
4. **网络深度**：可以调整`hidden_dims`增加或减少网络深度

## 验证训练效果

训练过程中会实时输出：
```
Epoch 1/100 - Train Loss: 0.123456, Train R²: 0.8500 | Val Loss: 0.134567, Val R²: 0.8400
```

- **R² > 0.8**：模型学习效果良好
- **R² > 0.9**：模型学习效果优秀
- **R² < 0.7**：可能需要调整网络结构或学习率

## 下一步

训练完Stage 1后，可以：
1. 检查模型性能（R²和损失）
2. 进行Stage 2训练（从无人机光谱预测盐分）
3. 进行联合微调（端到端优化）

