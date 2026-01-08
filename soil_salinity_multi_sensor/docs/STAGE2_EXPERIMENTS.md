# Stage 2 盐分反演对比实验

## 📋 实验概述

⭐ **这是主要的盐分反演实验流程**。

本实验脚本参照 `S2_salinity_inversion_plsr_vip.py` 的逻辑，实现5组对比实验，验证不同数据源对盐分反演效果的影响。

### 实验设计

| 实验 | 数据源 | 样点数量 | 样点筛选条件 |
|------|--------|----------|--------------|
| 实验1 | 原始UAV影像 | 30个 | 抽样编列开头为"FP" |
| 实验2 | 原始S2影像 | 108个 | 抽样编列开头不为"FP" |
| 实验3 | 原始L8影像 | 108个 | 抽样编列开头不为"FP" |
| 实验4 | S2校正后UAV光谱 | 108个 | 抽样编列开头不为"FP" |
| 实验5 | L8校正后UAV光谱 | 108个 | 抽样编列开头不为"FP" |

### 控制变量

- **特征数量**：所有实验使用30个特征（4个波段 + 26个光谱指数）
- **模型**：所有实验使用相同的模型集合（SVR, RandomForest, PLSR, GradientBoosting, XGBoost）
- **特征选择**：皮尔逊相关性 + RFE（阈值0.3，RFE选择15个特征）
- **评估指标**：R², RMSE, MAE, MedAE, MAPE, Adj_R², EVS, Pearson_r, Spearman_r

---

## 🚀 运行实验

### 方法1：使用main.py

```bash
python main.py --mode run_stage2_experiments \
    --data_config configs/data_config.yaml \
    --model_config configs/model_config.yaml
```

### 方法2：直接运行脚本

```bash
python evaluation/stage2_experiments.py
```

---

## 📊 实验流程

### 1. 数据准备

- **样点筛选**：根据"抽样编"列的前缀筛选样点
  - 实验1：选择"FP"开头的30个样点
  - 实验2-4：排除"FP"开头的108个样点

- **波段提取**：从对应栅格影像中提取样点位置的波段值
  - UAV：4个波段（G, R, REG, NIR）
  - S2：4个波段（B3, B4, B5, B8）
  - L8：3个波段（B3, B4, B5）+ REG填充为0

- **特征计算**：从波段值计算26个光谱指数
  - S1, S1REG, NDSI, NDSIREG
  - SI1, SI1REG, SI2, SI2REG, SI3, SI3REG
  - SIT, SITREG, Int1, Int1REG, Int2, Int2REG
  - NDVI, NDVIREG, SAVI, SAVIREG
  - BI, BIREG, IFe2O3, IFe2O3REG
  - DVI, DVIREG

### 2. 特征选择

- **皮尔逊相关性**：选择|r| >= 0.3的特征
- **RFE**：使用RandomForest估计器，选择15个特征
- **融合策略**：优先使用交集，如果交集<5则使用并集

### 3. 模型训练

训练以下模型（参照S2_salinity_inversion_plsr_vip.py）：

1. **SVR**：使用GridSearchCV调参
2. **RandomForest**：200棵树
3. **PLSR**：自动选择最优主成分数（1-20）
4. **GradientBoosting**：100棵树，最大深度5
5. **XGBoost**：100棵树（如果可用）

### 4. 模型评估

计算11项评估指标：
- MSE, RMSE, MAE, MedAE, MAPE
- R², Adj_R², EVS
- Pearson_r, Spearman_r

### 5. 结果输出

每个实验的输出目录：
```
outputs/stage2_experiments/
├── exp1_uav_original/
│   ├── experiment_results.xlsx
│   ├── scatter_svr.png
│   ├── scatter_randomforest.png
│   ├── scatter_plsr.png
│   ├── scatter_gradientboosting.png
│   └── scatter_xgboost.png
├── exp2_s2_original/
│   └── ...
├── exp3_l8_original/
│   └── ...
├── exp4_s2_corrected/
│   └── ...
├── exp5_l8_corrected/
│   └── ...
└── experiments_comparison.xlsx
```

---

## 📁 输出文件说明

### Excel报告（experiment_results.xlsx）

包含以下Sheet：

1. **模型评估**：所有模型的评估指标
2. **特征选择**：特征选择结果（皮尔逊相关系数、RFE排名、是否选中）
3. **元信息**：实验配置和统计信息

### 散点图

每个模型生成一个散点图，显示预测值vs真实值，包含1:1线。

### 对比报告（experiments_comparison.xlsx）

汇总所有4个实验的结果，便于对比分析。

---

## ⚙️ 配置说明

实验配置在 `evaluation/stage2_experiments.py` 中：

```python
# 特征选择参数
PEARSON_THRESHOLD = 0.3  # 皮尔逊相关系数阈值
RFE_N_FEATURES = 15      # RFE选择的特征数量
RFE_ESTIMATOR = "RandomForest"  # RFE估计器类型

# 数据划分
TEST_RATIO = 0.2         # 测试集比例
RANDOM_STATE = 42        # 随机种子

# 交叉验证
CV_FOLDS = 5             # 交叉验证折数
```

---

## 🔍 实验意义

### 实验1：原始UAV影像反演
- **目的**：建立基准性能（使用真实UAV数据）
- **预期**：应该获得最好的反演效果

### 实验2：原始S2影像反演
- **目的**：评估直接使用S2影像的反演效果
- **预期**：由于传感器差异，效果可能不如UAV

### 实验3：原始L8影像反演
- **目的**：评估直接使用L8影像的反演效果
- **预期**：由于缺少REG波段，效果可能较差

### 实验4：S2校正后UAV光谱反演
- **目的**：验证Stage 1对S2影像校正的有效性
- **预期**：如果Stage 1有效，应该接近实验1的效果，且优于实验2

### 实验5：L8校正后UAV光谱反演
- **目的**：验证Stage 1对L8影像校正的有效性
- **预期**：如果Stage 1有效，应该接近实验1的效果，且优于实验3

---

## 📈 结果分析建议

1. **对比实验1和实验4/5**：
   - 如果实验4接近实验1，说明Stage 1对S2的校正有效
   - 如果实验5接近实验1，说明Stage 1对L8的校正有效
   - 如果实验4/5明显优于实验2/3，说明校正改善了反演效果

2. **对比实验2和实验3**：
   - 评估S2和L8原始影像的反演能力差异
   - 分析L8缺少REG波段的影响

3. **模型性能对比**：
   - 找出最适合盐分反演的模型
   - 分析不同模型在不同数据源上的表现

---

## ⚠️ 注意事项

1. **数据准备**：
   - 确保样点Shapefile包含"抽样编"和"全盐量"列
   - 确保所有栅格影像路径正确

2. **实验4和5的说明**：
   - 实验4使用S2校正后的UAV光谱（s2_corrected_uav_spectrum.tif）
   - 实验5使用L8校正后的UAV光谱（l8_corrected_uav_spectrum.tif）
   - 两个实验独立运行，分别评估S2和L8的校正效果

3. **L8的REG波段**：
   - L8没有REG波段，实验3中REG值填充为0
   - 这会影响包含REG的指数计算

---

## 📚 相关文档

- [Stage 2 训练指南](STAGE2_TRAINING.md)
- [Stage 1 验证指南](STAGE1_VALIDATION.md)
- [运行说明](RUN_INSTRUCTIONS.md)

