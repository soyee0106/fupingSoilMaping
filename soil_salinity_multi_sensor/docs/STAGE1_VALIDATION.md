# Stage 1 验证实验指南

## 概述

本文档说明如何运行Stage 1的验证实验，验证模型是否真正实现了光谱解耦。

## 功能

### 1. 生成校正后的影像

使用训练好的Stage 1模型对整个影像进行预测，输出校正后的UAV光谱影像。

### 2. 验证实验

- **实验1：基础映射精度** - 计算S2和L8预测光谱与真实UAV光谱的R²和RMSE
- **实验2：跨传感器一致性** - 计算同一位置S2和L8预测光谱的差异
- **实验3：光谱曲线可视化** - 选取典型地物点，绘制真实UAV光谱、原始S2/L8光谱和预测光谱的对比曲线
- **实验4：偏差可视化** - 分析模型内部激活值，可视化S2和L8模式下的偏差差异

## 使用方法

### 1. 生成校正后的影像

```bash
python main.py --mode infer_stage1 \
    --data_config configs/data_config.yaml \
    --model_config configs/model_config.yaml
```

**输出：**
- `outputs/stage1_inference/s2_corrected_uav_spectrum.tif` - S2校正后的UAV光谱影像
- `outputs/stage1_inference/l8_corrected_uav_spectrum.tif` - L8校正后的UAV光谱影像

### 2. 运行验证实验

```bash
python main.py --mode validate_stage1 \
    --data_config configs/data_config.yaml \
    --model_config configs/model_config.yaml
```

**输出：**
- `outputs/stage1_validation/validation_results.json` - JSON格式的验证结果
- `outputs/stage1_validation/validation_report.txt` - 文本格式的验证报告
- `outputs/stage1_validation/spectral_curves/` - 光谱曲线可视化图
- `outputs/stage1_validation/bias_analysis/` - 偏差分析结果和可视化图

## 验证实验说明

### 实验1：基础映射精度

**目的：** 评估模型预测的UAV光谱与真实UAV光谱的匹配程度

**指标：**
- **R²（决定系数）**：越接近1越好，表示模型能解释的方差比例
- **RMSE（均方根误差）**：越小越好，表示预测误差

**输出：**
- S2和L8的总体R²和RMSE
- 每个波段（G, R, REG, NIR）的R²和RMSE

**解读：**
- R² > 0.8：模型预测精度高
- R² > 0.6：模型预测精度中等
- R² < 0.5：模型预测精度较低，需要改进

### 实验2：跨传感器一致性

**目的：** 验证模型是否真正实现了光谱解耦，即同一位置的不同传感器预测结果应该一致

**指标：**
- **欧氏距离**：S2和L8预测光谱之间的差异
- **各波段平均绝对差异**：每个波段的平均差异

**输出：**
- 平均欧氏距离（mean_euclidean_distance）
- 标准差（std_euclidean_distance）
- 中位数（median_euclidean_distance）
- 各波段平均绝对差异（mean_absolute_diff_per_band）

**解读：**
- 欧氏距离越小，说明S2和L8预测结果越一致
- 如果距离很小（< 0.1），说明模型成功解耦了传感器差异
- 如果距离很大（> 1.0），说明模型未能有效解耦传感器差异

### 实验3：光谱曲线可视化

**目的：** 直观展示模型预测效果，对比真实UAV光谱、原始卫星光谱和预测光谱

**方法：**
- 从地类验证点Shapefile（`originData/地类验证点.shp`）读取典型地物点位置
- 从对齐后的标准化影像中提取各点的光谱值
- 使用模型预测S2和L8输入对应的UAV光谱
- 绘制5条曲线：
  1. **真实UAV光谱（标准化后）** - 蓝色实线，带圆点标记
  2. **原始S2光谱（标准化后）** - 橙色实线，带方形标记
  3. **原始L8光谱（标准化后）** - 绿色实线，带三角标记
  4. **S2预测光谱（标准化后）** - 紫色虚线
  5. **L8预测光谱（标准化后）** - 灰色虚线

**输出：**
- `spectral_curves_all.png` - 所有点的光谱曲线（最多10个点）
- `spectral_curves_{地类}.png` - 按地类分组的光谱曲线（如果存在地类字段）

**解读：**
- 预测光谱应尽可能接近真实UAV光谱
- S2和L8预测光谱应尽可能一致（说明成功解耦了传感器差异）
- 如果存在地类字段，可以按地类（植被、裸土、水体等）分析不同地物的光谱特征

**注意：**
- 所有光谱值都是标准化后的值，在同一尺度下比较
- 如果地类验证点Shapefile不存在，实验3会跳过并给出警告

### 实验4：偏差可视化

**目的：** 深入分析模型内部机制，理解S2和L8模式下的偏差差异

**方法：**
- 从验证集中选择匹配的S2和L8样本对（同一地理位置）
- 提取模型第一层（ReLU激活后）的激活值
- 计算S2和L8模式下的激活值差异
- 分析输入波段对偏差的贡献

**输出：**
- `bias_visualization.png` - 包含4个子图：
  1. S2激活值分布
  2. L8激活值分布
  3. 激活值差异分布（S2 - L8）
  4. 各神经元平均差异
- `input_band_difference.png` - 各波段输入差异（S2 - L8）
- `bias_analysis_results.json` - 偏差分析统计结果

**解读：**
- 激活值差异分布应接近0，说明模型在不同传感器输入下产生了相似的内部表示
- 如果差异很大，说明模型未能有效解耦传感器差异
- 输入波段差异反映了S2和L8原始光谱的差异，这有助于理解模型需要学习的映射关系

**指标：**
- `mean_activation_diff` - 平均激活值差异
- `std_activation_diff` - 激活值差异标准差
- `mean_input_band_diff` - 各波段平均输入差异

## 结果示例

### 实验1结果示例

```
S2:
  总体R²: 0.5512
  总体RMSE: 0.4169
  各波段R²: [0.52, 0.48, 0.55, 0.60]
  各波段RMSE: [0.35, 0.42, 0.38, 0.50]

L8:
  总体R²: 0.4800
  总体RMSE: 0.5200
  各波段R²: [0.45, 0.50, 0.00, 0.52]
  各波段RMSE: [0.40, 0.45, 0.00, 0.55]
```

### 实验2结果示例

```
平均欧氏距离: 0.1234 ± 0.0567
中位数: 0.1100
各波段平均绝对差异: [0.05, 0.08, 0.12, 0.10]
```

## 注意事项

1. **数据标准化**：确保验证数据与训练数据使用相同的标准化方式
2. **模型路径**：默认使用 `outputs/models/stage1/best_model.pth`
3. **验证数据**：默认使用 `outputs/val_data.csv`
4. **标准化器**：需要从 `outputs/aligned_images/scalers/` 加载

## 故障排除

### 问题1：模型文件不存在
**解决：** 先运行训练命令生成模型
```bash
python main.py --mode train --stage stage1
```

### 问题2：验证数据不存在
**解决：** 先运行数据预处理
```bash
python main.py --mode preprocess
```

### 问题3：标准化器不存在
**解决：** 确保在预处理时启用了标准化（`normalize: true`）

## 下一步

验证实验完成后，可以：
1. 分析结果，评估模型性能
2. 如果结果不理想，调整模型超参数或网络结构
3. 进行Stage 2训练（盐分反演）

