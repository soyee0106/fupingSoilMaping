# Stage 1 完整功能指南

## 功能概述

Stage 1实现了光谱解耦网络（传感器偏差解码器），并提供了完整的训练、推理和验证功能。

## 已实现功能

### 1. 模型训练 ✅
- 使用MLP（多层感知机）架构
- 支持S2和L8两种传感器
- 自动处理波段缺失（L8的REG波段）
- 实时监控R²和MSE损失

### 2. 影像推理 ✅
- 对整个影像进行预测
- 输出校正后的UAV光谱影像
- 支持S2和L8两种输入

### 3. 验证实验 ✅
- **实验1：基础映射精度** - 评估预测精度
- **实验2：跨传感器一致性** - 验证光谱解耦效果

## 使用流程

### 步骤1：数据预处理
```bash
python main.py --mode preprocess --data_config configs/data_config.yaml
```

### 步骤2：训练Stage 1模型
```bash
python main.py --mode train \
    --data_config configs/data_config.yaml \
    --model_config configs/model_config.yaml \
    --stage stage1
```

### 步骤3：生成校正后的影像
```bash
python main.py --mode infer_stage1 \
    --data_config configs/data_config.yaml \
    --model_config configs/model_config.yaml
```

### 步骤4：运行验证实验
```bash
python main.py --mode validate_stage1 \
    --data_config configs/data_config.yaml \
    --model_config configs/model_config.yaml
```

## 输出文件

### 训练输出
- `outputs/models/stage1/best_model.pth` - 最佳模型
- `outputs/models/stage1/stage1_final_model.pth` - 最终模型

### 推理输出
- `outputs/stage1_inference/s2_corrected_uav_spectrum.tif` - S2校正后的UAV光谱影像（4波段）
- `outputs/stage1_inference/l8_corrected_uav_spectrum.tif` - L8校正后的UAV光谱影像（4波段）

### 验证输出
- `outputs/stage1_validation/validation_results.json` - JSON格式结果
- `outputs/stage1_validation/validation_report.txt` - 文本格式报告

## 验证实验解读

### 实验1：基础映射精度

**指标说明：**
- **R²（决定系数）**：
  - R² = 1.0：完美预测
  - R² > 0.8：优秀
  - R² > 0.6：良好
  - R² > 0.5：中等
  - R² < 0.5：需要改进

- **RMSE（均方根误差）**：
  - 越小越好
  - 在标准化数据上，RMSE < 1.0 通常表示良好性能

**期望结果：**
- S2的R²应该 > 0.5（当前约0.55）
- L8的R²可能略低（因为缺少REG波段）
- 各波段的R²应该相对均衡

### 实验2：跨传感器一致性

**指标说明：**
- **欧氏距离**：
  - 距离 < 0.1：高度一致，解耦成功
  - 距离 < 0.3：较好一致
  - 距离 > 0.5：一致性较差，解耦效果不佳

**期望结果：**
- 如果模型成功解耦，同一位置的S2和L8预测应该非常接近
- 平均欧氏距离应该 < 0.3（在标准化数据上）

## 验证光谱解耦成功的标准

1. **实验1结果良好**：
   - S2和L8的R²都 > 0.5
   - RMSE在合理范围内

2. **实验2结果良好**：
   - 平均欧氏距离 < 0.3
   - 说明不同传感器预测结果一致

3. **两个实验都通过**：
   - 说明模型既准确预测了UAV光谱，又成功解耦了传感器差异

## 常见问题

### Q1: R²为负数怎么办？
**A:** 说明模型性能很差，可能原因：
- 数据未正确标准化
- 模型训练不充分
- 网络结构不合适

### Q2: 跨传感器一致性很差怎么办？
**A:** 说明模型未能有效解耦传感器差异，可能原因：
- 训练数据不足
- 模型容量不够
- 需要调整网络结构

### Q3: 如何提高性能？
**A:** 可以尝试：
- 增加训练轮数
- 调整学习率
- 增加网络层数或神经元数
- 使用更多训练数据

## 下一步

验证通过后，可以：
1. 进行Stage 2训练（盐分反演）
2. 进行联合微调（端到端优化）
3. 应用到实际区域进行盐分反演

