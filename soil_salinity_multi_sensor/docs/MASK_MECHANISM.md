# 波段掩码机制说明

## 概述

本系统实现了**波段掩码机制**来处理不同传感器之间的波段不对称性问题。当某些传感器（如Landsat-8）缺少特定波段（如REG波段）时，系统能够智能处理这种缺失，而不是简单地进行波段一一映射。

## 核心设计

### 1. 数据层面

#### 统一波段格式
所有卫星数据统一为 **G, R, REG, NIR** 格式：
- **Sentinel-2**: 有B, G, R, NIR，映射为 G, R, REG(填充0), NIR
- **Landsat-8**: 有B, G, R, NIR，映射为 G, R, REG(填充0), NIR

#### 掩码向量生成
为每个样本生成掩码向量，标记哪些波段有效：
- `1` = 波段有效
- `0` = 波段缺失（已填充0）

**示例**：
- S2掩码: `[1, 1, 0, 1]` (G有效, R有效, REG缺失, NIR有效)
- L8掩码: `[1, 1, 0, 1]` (G有效, R有效, REG缺失, NIR有效)

### 2. 模型层面

#### 阶段一网络（SensorBiasDecoder）

**输入**：
- `satellite_bands`: 卫星波段特征 (batch_size, 4) - 统一格式G, R, REG, NIR
- `spectral_indices`: 光谱指数 (batch_size, 26)
- `sensor_onehot`: 传感器标签 (batch_size, 2)
- `band_mask`: 掩码向量 (batch_size, 4) - **新增**

**处理逻辑**：
1. 使用掩码对缺失波段进行屏蔽：`masked_bands = satellite_bands * band_mask`
2. 将掩码向量作为额外特征输入网络
3. 网络学习在REG缺失时，仅基于G, R, NIR进行光谱校正

**配置参数**：
```yaml
stage1:
  n_satellite_bands: 4
  n_spectral_indices: 26
  n_sensors: 2
  n_mask_bits: 4  # 掩码位数
  use_mask: true  # 启用掩码机制
```

### 3. 数据预处理流程

#### 提取卫星特征
```python
# S2数据处理
s2_bands = {
    'G': s2_bands_raw['S2_B3'],  # 绿光
    'R': s2_bands_raw['S2_B4'],  # 红光
    'REG': np.zeros(len(s2_bands_raw)),  # 缺失，填充0
    'NIR': s2_bands_raw['S2_B8'],  # 近红外
}
s2_band_mask = [[1, 1, 0, 1]]  # 掩码向量

# L8数据处理
l8_bands = {
    'G': l8_bands_raw['L8_B3'],
    'R': l8_bands_raw['L8_B4'],
    'REG': np.zeros(len(l8_bands_raw)),  # 缺失，填充0
    'NIR': l8_bands_raw['L8_B5'],
}
l8_band_mask = [[1, 1, 0, 1]]  # 掩码向量
```

#### 掩码存储
掩码向量以字符串形式存储在DataFrame的`band_mask`列中：
```python
df['band_mask'] = ['1,1,0,1', '1,1,0,1', ...]
```

### 4. 训练流程

#### 数据加载器
数据加载器需要返回掩码向量：
```python
batch = (
    satellite_bands,      # (batch_size, 4)
    spectral_indices,      # (batch_size, 26)
    sensor_onehot,         # (batch_size, 2)
    uav_bands_true,       # (batch_size, 4)
    band_mask             # (batch_size, 4) - 新增
)
```

#### 训练函数
```python
# 前向传播时传入掩码
uav_bands_pred = model(
    satellite_bands,
    spectral_indices,
    sensor_onehot,
    band_mask  # 传入掩码
)
```

### 5. 工具函数

#### 解析掩码
```python
from data_preprocessing import parse_band_mask, get_band_mask_from_df

# 从字符串解析
mask = parse_band_mask("1,1,0,1")  # 返回 [1, 1, 0, 1]

# 从DataFrame提取
masks = get_band_mask_from_df(df, mask_column='band_mask')
```

## 使用示例

### 1. 数据预处理
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

# DataFrame中会包含band_mask列
print(df['band_mask'].head())
# 0    1,1,0,1
# 1    1,1,0,1
# ...
```

### 2. 模型训练
```python
from models import SensorBiasDecoder
from training import train_stage1_decoder

# 创建模型（启用掩码）
model = SensorBiasDecoder(
    n_satellite_bands=4,
    n_spectral_indices=26,
    n_sensors=2,
    n_mask_bits=4,
    use_mask=True  # 启用掩码机制
)

# 训练时数据加载器需要提供掩码
train_stage1_decoder(model, train_loader, val_loader, ...)
```

### 3. 模型推理
```python
# 准备掩码向量
band_mask = torch.tensor([[1, 1, 0, 1]])  # L8数据

# 前向传播
outputs = model(
    satellite_bands,
    spectral_indices,
    sensor_onehot,
    band_mask
)
```

## 优势

1. **智能处理缺失波段**：模型能够学习在REG缺失时，仅基于其他有效波段进行校正
2. **统一数据格式**：所有传感器数据统一为相同格式，简化处理流程
3. **显式缺失标记**：掩码向量明确告知模型哪些波段有效，哪些缺失
4. **灵活扩展**：可以轻松扩展到其他缺失波段的情况

## 注意事项

1. **掩码一致性**：确保掩码向量与波段数据的对应关系正确
2. **损失计算**：可以考虑使用掩码加权损失，只对有效波段计算损失
3. **数据验证**：预处理后检查掩码向量是否正确生成

## 未来改进

1. **掩码加权损失**：在损失计算中考虑掩码，只对有效波段计算损失
2. **自适应掩码**：根据数据质量动态生成掩码
3. **多波段缺失**：扩展到处理多个波段缺失的情况

