# 运行指令说明

## 数据预处理模式（生成对齐影像和配对数据）

### 1. 密集映射模式（像素到像素对齐）

此模式会将UAV影像聚合对齐到Sentinel-2，并生成：
- 对齐后的卫星影像（GeoTIFF）
- 对齐后的UAV影像（GeoTIFF）
- 像元配对数据表（CSV）

**运行命令：**

```powershell
# 激活conda环境
conda activate torch38

# 进入项目目录
cd "D:\git\1230富平盐分反演\soil_salinity_multi_sensor"

# 运行预处理（密集映射模式）
python main.py --mode preprocess --data_config configs/data_config.yaml
```

**输出文件：**
- `outputs/aligned_images/satellite_aligned.tif` - 对齐后的卫星影像
- `outputs/aligned_images/uav_aligned.tif` - 对齐后的UAV影像
- `outputs/aligned_images/pixel_pairs.csv` - 像元配对数据表
- `outputs/preprocessed_data.csv` - 预处理后的完整数据（包含光谱指数）

### 2. 样点提取模式

此模式从样点位置提取光谱值，用于有盐分值标签的训练数据。

**运行命令：**

```powershell
python main.py --mode preprocess --data_config configs/data_config.yaml
```

**注意：** 需要在 `data_config.yaml` 中设置 `preprocessing.dense_mapping.enabled: false`

**输出文件：**
- `outputs/preprocessed_data.csv` - 包含样点光谱值和盐分值的数据

## 影像对齐模式（独立对齐功能）

### UAV影像聚合对齐到S2

```powershell
python main.py --mode align --data_config configs/data_config.yaml --stage uav
```

**输出文件：**
- `outputs/aligned_images/uav_aligned_to_s2.tif`
- `outputs/aligned_images/uav_s2_pixel_pairs.csv`

### Landsat-8重采样对齐到S2

```powershell
python main.py --mode align --data_config configs/data_config.yaml --stage l8
```

**输出文件：**
- `outputs/aligned_images/l8_aligned_to_s2.tif`
- `outputs/aligned_images/l8_s2_pixel_pairs.csv`

## 模型训练模式

### 阶段一训练（学习卫星→无人机映射）

```powershell
python main.py --mode train --data_config configs/data_config.yaml --model_config configs/model_config.yaml --stage stage1
```

### 阶段二训练（学习无人机→盐分值映射）

```powershell
python main.py --mode train --data_config configs/data_config.yaml --model_config configs/model_config.yaml --stage stage2
```

### 联合微调（完整模型）

```powershell
python main.py --mode train --data_config configs/data_config.yaml --model_config configs/model_config.yaml --stage full
```

## 配置文件说明

### data_config.yaml

关键配置项：

```yaml
# 数据路径
data_paths:
  s2_raster: "路径/Sentinel-2影像.tif"
  l8_raster: "路径/Landsat-8影像.tif"
  uav_raster: "路径/UAV影像.tif"
  samples_shapefile: "路径/样点.shp"

# 密集映射配置
preprocessing:
  dense_mapping:
    enabled: true  # true=密集映射模式，false=样点提取模式
    align_to: "satellite"  # 对齐到卫星网格
    uav_nodata: 65535
    s2_nodata: 0
```

## 注意事项

1. **密集映射模式**：用于学习像素到像素的光谱映射关系，不需要盐分值标签
2. **样点提取模式**：需要样点Shapefile和盐分值列，用于有监督训练
3. **输出目录**：所有输出文件会自动创建在 `outputs/` 目录下
4. **内存占用**：大影像处理时注意内存使用情况
5. **CRS一致性**：代码会自动处理CRS不一致的情况（自动重投影）

## 常见问题

### Q: 如何处理不同分辨率的影像？
A: 代码会自动将高分辨率影像聚合到目标分辨率，使用块均值方法。

### Q: 如何指定要提取的波段？
A: 在 `data_config.yaml` 中配置 `bands.satellite.s2.band_mapping` 或使用 `satellite_band_indices` 参数。

### Q: 输出影像的nodata值是什么？
A: 默认使用 `-9999.0`（float32类型），可以在代码中修改。

