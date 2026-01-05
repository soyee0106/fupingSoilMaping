import numpy as np
import rasterio
import matplotlib.pyplot as plt
from scipy import ndimage, signal
import cv2
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ====================== 1. 读取无人机高分辨率图像 ======================
def load_uav_image(uav_path):
    """读取无人机多光谱图像"""
    print(f"读取无人机图像: {uav_path}")
    
    try:
        with rasterio.open(uav_path) as src:
            # 读取所有波段
            img = src.read()
            print(f"图像信息:")
            print(f"  波段数: {img.shape[0]}")
            print(f"  高度: {img.shape[1]}")
            print(f"  宽度: {img.shape[2]}")
            print(f"  数据类型: {img.dtype}")
            print(f"  投影: {src.crs}")
            print(f"  地理变换: {src.transform}")
            
            # 显示波段名称（如果有）
            if hasattr(src, 'descriptions') and src.descriptions[0]:
                print(f"  波段描述: {src.descriptions}")
            
            # 转换为浮点型并归一化
            img_float = img.astype(np.float32)
            
            # 对每个波段进行归一化到0-1范围
            for i in range(img_float.shape[0]):
                band = img_float[i]
                min_val = np.min(band)
                max_val = np.max(band)
                if max_val > min_val:
                    img_float[i] = (band - min_val) / (max_val - min_val)
            
            return img_float, src
        
    except Exception as e:
        print(f"读取图像失败: {e}")
        
        # 如果读取失败，创建一个模拟图像用于演示
        print("创建模拟图像用于演示...")
        hr_img = np.zeros((4, 512, 512), dtype=np.float32)
        
        # 创建4个波段：蓝、绿、红、近红外
        for band_idx in range(4):
            # 添加地形纹理
            x = np.linspace(-2, 2, 512)
            y = np.linspace(-2, 2, 512)
            X, Y = np.meshgrid(x, y)
            Z = np.sin(X*3) * np.cos(Y*3) * 0.3 + np.sin(X*5) * np.cos(Y*5) * 0.2
            
            # 添加随机农田纹理
            texture = np.random.randn(512, 512) * 0.1
            texture = ndimage.gaussian_filter(texture, sigma=10)
            
            # 添加一些农田边界
            boundaries = np.zeros((512, 512))
            for i in range(10):
                center_x = np.random.randint(100, 412)
                center_y = np.random.randint(100, 412)
                radius = np.random.randint(30, 80)
                cv2.circle(boundaries, (center_x, center_y), radius, 1, 3)
            
            # 组合
            hr_img[band_idx] = Z + texture * 0.5 + boundaries * 0.3
            
            # 归一化
            min_val = np.min(hr_img[band_idx])
            max_val = np.max(hr_img[band_idx])
            hr_img[band_idx] = (hr_img[band_idx] - min_val) / (max_val - min_val)
        
        print(f"创建模拟图像: {hr_img.shape}")
        return hr_img, None

# ====================== 2. 定义退化模型 ======================
def create_gaussian_psf(size=15, sigma=2.0):
    """创建高斯点扩散函数(PSF)"""
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    kernel = kernel / np.sum(kernel)  # 归一化
    return kernel

def create_motion_psf(size=15, angle=30):
    """创建运动模糊点扩散函数"""
    psf = np.zeros((size, size))
    center = size // 2
    length = size // 3
    
    # 创建线形PSF
    angle_rad = np.deg2rad(angle)
    dx = int(np.cos(angle_rad) * length)
    dy = int(np.sin(angle_rad) * length)
    
    cv2.line(psf, (center - dx, center - dy), (center + dx, center + dy), 1, 1)
    psf = psf / np.sum(psf)  # 归一化
    return psf

def physical_degradation(hr_img, scale_factor=200, psf_type='gaussian', noise_level=0.02):
    """
    物理退化模型: PSF模糊 + 下采样 + 噪声
    
    参数:
        hr_img: 高分辨率图像 (C, H, W) 或 (H, W)
        scale_factor: 下采样比例
        psf_type: PSF类型 ('gaussian', 'motion', 'both')
        noise_level: 噪声水平
    """
    # 确保是3D数组 (波段, 高, 宽)
    if len(hr_img.shape) == 2:
        hr_img = hr_img[np.newaxis, :, :]
    
    num_bands, h_hr, w_hr = hr_img.shape
    
    # 计算下采样后的尺寸
    h_lr = h_hr // scale_factor
    w_lr = w_hr // scale_factor
    
    lr_img = np.zeros((num_bands, h_lr, w_lr), dtype=np.float32)
    
    # 创建PSF
    if psf_type == 'gaussian':
        psf = create_gaussian_psf(size=min(15, scale_factor*4), sigma=scale_factor/2)
    elif psf_type == 'motion':
        psf = create_motion_psf(size=min(15, scale_factor*4), angle=30)
    elif psf_type == 'both':
        psf1 = create_gaussian_psf(size=min(15, scale_factor*4), sigma=scale_factor/2)
        psf2 = create_motion_psf(size=min(15, scale_factor*4), angle=30)
        psf = signal.convolve2d(psf1, psf2, mode='same')
        psf = psf / np.sum(psf)
    else:
        raise ValueError(f"未知的PSF类型: {psf_type}")
    
    print(f"使用PSF类型: {psf_type}, 大小: {psf.shape}, 下采样因子: {scale_factor}")
    
    # 对每个波段应用退化
    for band_idx in range(num_bands):
        band_hr = hr_img[band_idx]
        
        # 1. 应用PSF模糊
        band_blurred = signal.convolve2d(band_hr, psf, mode='same', boundary='symm')
        
        # 2. 下采样 (使用平均池化模拟传感器积分)
        band_lr = np.zeros((h_lr, w_lr))
        for i in range(h_lr):
            for j in range(w_lr):
                i_start = i * scale_factor
                j_start = j * scale_factor
                patch = band_blurred[i_start:i_start+scale_factor, j_start:j_start+scale_factor]
                band_lr[i, j] = np.mean(patch)
        
        # 3. 添加噪声 (模拟传感器噪声)
        noise = np.random.normal(0, noise_level, band_lr.shape)
        band_lr += noise
        
        # 4. 裁剪到合理范围
        band_lr = np.clip(band_lr, 0, 1)
        
        lr_img[band_idx] = band_lr
    
    return lr_img, psf

def simple_resampling(hr_img, scale_factor=4, method='bicubic'):
    """
    直接重采样: 仅下采样，无物理退化
    
    参数:
        hr_img: 高分辨率图像 (C, H, W) 或 (H, W)
        scale_factor: 下采样比例
        method: 重采样方法 ('nearest', 'bilinear', 'bicubic', 'average')
    """
    # 确保是3D数组 (波段, 高, 宽)
    if len(hr_img.shape) == 2:
        hr_img = hr_img[np.newaxis, :, :]
    
    num_bands, h_hr, w_hr = hr_img.shape
    
    # 计算下采样后的尺寸
    h_lr = h_hr // scale_factor
    w_lr = w_hr // scale_factor
    
    lr_img = np.zeros((num_bands, h_lr, w_lr), dtype=np.float32)
    
    print(f"直接重采样方法: {method}, 下采样因子: {scale_factor}")
    
    # 对每个波段应用重采样
    for band_idx in range(num_bands):
        band_hr = hr_img[band_idx]
        
        if method == 'nearest':
            lr_img[band_idx] = cv2.resize(band_hr, (w_lr, h_lr), interpolation=cv2.INTER_NEAREST)
        elif method == 'bilinear':
            lr_img[band_idx] = cv2.resize(band_hr, (w_lr, h_lr), interpolation=cv2.INTER_LINEAR)
        elif method == 'bicubic':
            lr_img[band_idx] = cv2.resize(band_hr, (w_lr, h_lr), interpolation=cv2.INTER_CUBIC)
        elif method == 'average':
            # 平均池化
            band_lr = np.zeros((h_lr, w_lr))
            for i in range(h_lr):
                for j in range(w_lr):
                    i_start = i * scale_factor
                    j_start = j * scale_factor
                    patch = band_hr[i_start:i_start+scale_factor, j_start:j_start+scale_factor]
                    band_lr[i, j] = np.mean(patch)
            lr_img[band_idx] = band_lr
        else:
            raise ValueError(f"未知的重采样方法: {method}")
    
    return lr_img

# ====================== 3. 质量评估指标 ======================
def calculate_metrics(hr_img, lr_img_upscaled, scale_factor=4):
    """
    计算图像质量指标
    
    参数:
        hr_img: 原始高分辨率图像 (H, W) 或 (C, H, W)
        lr_img_upscaled: 上采样后的低分辨率图像，与hr_img相同尺寸
        scale_factor: 下采样比例
    
    返回:
        包含PSNR, SSIM, RMSE, ERGAS的字典
    """
    # 如果是多波段图像，取第一个波段计算指标
    if len(hr_img.shape) == 3:
        hr_band = hr_img[0]
        lr_band = lr_img_upscaled[0]
    else:
        hr_band = hr_img
        lr_band = lr_img_upscaled
    
    # 计算RMSE
    rmse = np.sqrt(np.mean((hr_band - lr_band) ** 2))
    
    # 计算PSNR
    max_pixel = 1.0
    if rmse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(max_pixel / rmse)
    
    # 计算SSIM (简化版本)
    def calculate_ssim(img1, img2):
        C1 = (0.01 * 1) ** 2
        C2 = (0.03 * 1) ** 2
        
        mu1 = ndimage.gaussian_filter(img1, 1.5)
        mu2 = ndimage.gaussian_filter(img2, 1.5)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = ndimage.gaussian_filter(img1 ** 2, 1.5) - mu1_sq
        sigma2_sq = ndimage.gaussian_filter(img2 ** 2, 1.5) - mu2_sq
        sigma12 = ndimage.gaussian_filter(img1 * img2, 1.5) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return np.mean(ssim_map)
    
    ssim = calculate_ssim(hr_band, lr_band)
    
    # 计算ERGAS (用于多光谱图像评估)
    if len(hr_img.shape) == 3:
        num_bands = hr_img.shape[0]
        ergas_sum = 0
        for b in range(num_bands):
            mse_band = np.mean((hr_img[b] - lr_img_upscaled[b]) ** 2)
            mean_hr = np.mean(hr_img[b])
            if mean_hr > 0:
                ergas_sum += mse_band / (mean_hr ** 2)
        
        ergas = 100 * (1 / scale_factor) * np.sqrt(ergas_sum / num_bands)
    else:
        ergas = None
    
    return {
        'PSNR': psnr,
        'SSIM': ssim,
        'RMSE': rmse,
        'ERGAS': ergas
    }

# ====================== 4. 可视化函数 ======================
def plot_comparison(hr_img, resampled_img, physical_img, 
                   resampled_metrics, physical_metrics,
                   scale_factor=4, band_idx=0):
    """
    绘制对比结果
    
    参数:
        hr_img: 原始高分辨率图像 (C, H, W)
        resampled_img: 直接重采样结果 (C, H/s, W/s)
        physical_img: 物理退化结果 (C, H/s, W/s)
        resampled_metrics: 直接重采样质量指标
        physical_metrics: 物理退化质量指标
        scale_factor: 下采样比例
        band_idx: 要显示的波段索引
    """
    # 提取指定波段
    hr_band = hr_img[band_idx] if len(hr_img.shape) == 3 else hr_img
    resampled_band = resampled_img[band_idx] if len(resampled_img.shape) == 3 else resampled_img
    physical_band = physical_img[band_idx] if len(physical_img.shape) == 3 else physical_img
    
    # 将低分辨率图像上采样到原始尺寸以便对比
    h_hr, w_hr = hr_band.shape
    h_lr, w_lr = resampled_band.shape
    
    resampled_upscaled = cv2.resize(resampled_band, (w_hr, h_hr), interpolation=cv2.INTER_CUBIC)
    physical_upscaled = cv2.resize(physical_band, (w_hr, h_hr), interpolation=cv2.INTER_CUBIC)
    
    # 创建图像列表和标题
    images = [
        hr_band,
        resampled_upscaled,
        physical_upscaled,
        cv2.resize(resampled_band, (w_hr, h_hr), interpolation=cv2.INTER_NEAREST),  # 最近邻放大，查看原始LR像素
        cv2.resize(physical_band, (w_hr, h_hr), interpolation=cv2.INTER_NEAREST)   # 最近邻放大，查看原始LR像素
    ]
    
    titles = [
        f'原始高分辨率图像\n尺寸: {h_hr}x{w_hr}',
        f'直接重采样 (双三次)\nPSNR: {resampled_metrics["PSNR"]:.2f}dB, SSIM: {resampled_metrics["SSIM"]:.3f}',
        f'物理退化 (PSF+下采样)\nPSNR: {physical_metrics["PSNR"]:.2f}dB, SSIM: {physical_metrics["SSIM"]:.3f}',
        f'直接重采样 (最近邻放大显示)\nLR尺寸: {h_lr}x{w_lr}',
        f'物理退化 (最近邻放大显示)\nLR尺寸: {h_lr}x{w_lr}'
    ]
    
    # 绘制图像
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for idx in range(5):
        axes[idx].imshow(images[idx], cmap='gray')
        axes[idx].set_title(titles[idx], fontsize=11)
        axes[idx].axis('off')
    
    # 第六个子图显示质量指标对比
    axes[5].axis('off')
    metrics_text = f"质量指标对比 (波段 {band_idx+1}):\n\n"
    metrics_text += f"直接重采样:\n"
    metrics_text += f"  PSNR: {resampled_metrics['PSNR']:.2f} dB\n"
    metrics_text += f"  SSIM: {resampled_metrics['SSIM']:.3f}\n"
    metrics_text += f"  RMSE: {resampled_metrics['RMSE']:.4f}\n"
    if resampled_metrics['ERGAS'] is not None:
        metrics_text += f"  ERGAS: {resampled_metrics['ERGAS']:.2f}\n"
    
    metrics_text += f"\n物理退化:\n"
    metrics_text += f"  PSNR: {physical_metrics['PSNR']:.2f} dB\n"
    metrics_text += f"  SSIM: {physical_metrics['SSIM']:.3f}\n"
    metrics_text += f"  RMSE: {physical_metrics['RMSE']:.4f}\n"
    if physical_metrics['ERGAS'] is not None:
        metrics_text += f"  ERGAS: {physical_metrics['ERGAS']:.2f}\n"
    
    metrics_text += f"\n下采样因子: {scale_factor}"
    
    axes[5].text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center')
    
    plt.suptitle(f'直接重采样 vs 物理退化效果对比 (波段 {band_idx+1})', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.show()
    
    return resampled_upscaled, physical_upscaled

def plot_psf_and_degradation_process(psf, hr_patch, resampled_patch, physical_patch, scale_factor):
    """绘制PSF和退化过程"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 显示PSF
    axes[0, 0].imshow(psf, cmap='hot')
    axes[0, 0].set_title(f'点扩散函数(PSF)\n大小: {psf.shape[0]}x{psf.shape[1]}')
    axes[0, 0].axis('off')
    
    # 显示PSF的3D视图
    from mpl_toolkits.mplot3d import Axes3D
    ax = fig.add_subplot(2, 3, 2, projection='3d')
    x = np.arange(psf.shape[1])
    y = np.arange(psf.shape[0])
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, psf, cmap='hot', linewidth=0, antialiased=True)
    ax.set_title('PSF 3D视图')
    ax.set_zlabel('强度')
    
    # 显示原始高分辨率图像块
    axes[0, 2].imshow(hr_patch, cmap='gray')
    axes[0, 2].set_title('原始HR图像块')
    axes[0, 2].axis('off')
    
    # 显示直接重采样结果
    axes[1, 0].imshow(resampled_patch, cmap='gray')
    axes[1, 0].set_title('直接重采样结果')
    axes[1, 0].axis('off')
    
    # 显示物理退化结果
    axes[1, 1].imshow(physical_patch, cmap='gray')
    axes[1, 1].set_title('物理退化结果')
    axes[1, 1].axis('off')
    
    # 显示差异图
    diff = np.abs(resampled_patch - physical_patch)
    axes[1, 2].imshow(diff, cmap='hot')
    axes[1, 2].set_title('两者差异 (热力图)')
    axes[1, 2].axis('off')
    
    plt.suptitle(f'退化过程分析 (下采样因子: {scale_factor})', fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_spectral_profiles(hr_img, resampled_img, physical_img, 
                          resampled_upscaled, physical_upscaled,
                          band_names=None):
    """绘制光谱剖面图对比"""
    if band_names is None:
        band_names = [f'波段{i+1}' for i in range(hr_img.shape[0])]
    
    # 选择一个像素位置查看光谱曲线
    h, w = hr_img.shape[1], hr_img.shape[2]
    px, py = h//2, w//2  # 中心像素
    
    # 提取光谱曲线
    hr_profile = hr_img[:, px, py]
    resampled_profile = resampled_img[:, px//4, py//4]  # 注意：下采样后位置
    physical_profile = physical_img[:, px//4, py//4]
    resampled_up_profile = resampled_upscaled[:, px, py]
    physical_up_profile = physical_upscaled[:, px, py]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 绘制原始光谱曲线
    axes[0].plot(band_names, hr_profile, 'b-', linewidth=3, label='原始HR')
    axes[0].plot(band_names, resampled_profile, 'g--', linewidth=2, label='直接重采样(LR)')
    axes[0].plot(band_names, physical_profile, 'r--', linewidth=2, label='物理退化(LR)')
    axes[0].set_xlabel('波段')
    axes[0].set_ylabel('反射率/像素值')
    axes[0].set_title('低分辨率图像光谱曲线对比')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 绘制上采样后光谱曲线
    axes[1].plot(band_names, hr_profile, 'b-', linewidth=3, label='原始HR')
    axes[1].plot(band_names, resampled_up_profile, 'g:', linewidth=2, label='直接重采样(上采样后)')
    axes[1].plot(band_names, physical_up_profile, 'r:', linewidth=2, label='物理退化(上采样后)')
    axes[1].set_xlabel('波段')
    axes[1].set_ylabel('反射率/像素值')
    axes[1].set_title('上采样后光谱曲线对比')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(f'像素位置({px}, {py})的光谱剖面对比', fontsize=14)
    plt.tight_layout()
    plt.show()

# ====================== 5. 主程序 ======================
def main():
    # 设置参数
    uav_path = r"D:\富平星机光谱融合反演\数据\multi_G_R_REG_NIR.tif"
    scale_factor = 4  # 下采样比例
    band_to_display = 0  # 要显示的波段 (0-based)
    psf_type = 'gaussian'  # PSF类型: 'gaussian', 'motion', 'both'
    
    print("="*70)
    print("高分辨率图像重采样 vs 物理退化效果对比")
    print("="*70)
    
    # 1. 加载无人机图像
    hr_img, src = load_uav_image(uav_path)
    
    # 如果没有成功加载，使用模拟图像
    if hr_img is None:
        print("无法加载图像，退出程序")
        return
    
    # 2. 应用直接重采样
    print("\n" + "-"*50)
    print("应用直接重采样...")
    resampled_img = simple_resampling(hr_img, scale_factor=scale_factor, method='bicubic')
    
    # 3. 应用物理退化
    print("\n" + "-"*50)
    print("应用物理退化...")
    physical_img, psf = physical_degradation(
        hr_img, 
        scale_factor=scale_factor, 
        psf_type=psf_type,
        noise_level=0.02
    )
    
    # 4. 计算质量指标
    print("\n" + "-"*50)
    print("计算质量指标...")
    
    # 将低分辨率图像上采样到原始尺寸以便计算指标
    h_hr, w_hr = hr_img.shape[1], hr_img.shape[2]
    
    resampled_upscaled = np.zeros_like(hr_img)
    physical_upscaled = np.zeros_like(hr_img)
    
    for b in range(hr_img.shape[0]):
        resampled_band = resampled_img[b]
        physical_band = physical_img[b]
        
        resampled_upscaled[b] = cv2.resize(resampled_band, (w_hr, h_hr), interpolation=cv2.INTER_CUBIC)
        physical_upscaled[b] = cv2.resize(physical_band, (w_hr, h_hr), interpolation=cv2.INTER_CUBIC)
    
    # 计算指标
    resampled_metrics = calculate_metrics(hr_img, resampled_upscaled, scale_factor)
    physical_metrics = calculate_metrics(hr_img, physical_upscaled, scale_factor)
    
    # 5. 显示结果
    print("\n" + "-"*50)
    print("显示对比结果...")
    
    # 主对比图
    resampled_up, physical_up = plot_comparison(
        hr_img, resampled_img, physical_img,
        resampled_metrics, physical_metrics,
        scale_factor=scale_factor,
        band_idx=band_to_display
    )
    
    # 提取图像块用于详细分析
    patch_size = 100
    h, w = hr_img.shape[1], hr_img.shape[2]
    patch_start_h = h//2 - patch_size//2
    patch_start_w = w//2 - patch_size//2
    
    hr_patch = hr_img[band_to_display, 
                     patch_start_h:patch_start_h+patch_size, 
                     patch_start_w:patch_start_w+patch_size]
    
    resampled_patch = resampled_up[band_to_display, 
                                  patch_start_h:patch_start_h+patch_size, 
                                  patch_start_w:patch_start_w+patch_size]
    
    physical_patch = physical_up[band_to_display, 
                                patch_start_h:patch_start_h+patch_size, 
                                patch_start_w:patch_start_w+patch_size]
    
    # 显示PSF和退化过程
    plot_psf_and_degradation_process(
        psf, hr_patch, resampled_patch, physical_patch, scale_factor
    )
    
    # 如果是多光谱图像，显示光谱剖面
    if hr_img.shape[0] > 1:
        print("\n" + "-"*50)
        print("显示光谱剖面对比...")
        
        # 假设波段名称
        if hr_img.shape[0] >= 4:
            band_names = ['蓝', '绿', '红', '近红外'][:hr_img.shape[0]]
        else:
            band_names = [f'波段{i+1}' for i in range(hr_img.shape[0])]
        
        plot_spectral_profiles(
            hr_img, resampled_img, physical_img,
            resampled_upscaled, physical_upscaled,
            band_names=band_names
        )
    
    # 6. 总结分析
    print("\n" + "="*70)
    print("总结分析:")
    print("="*70)
    print("""
    直接重采样 vs 物理退化对比:

    1. 直接重采样 (Simple Resampling):
       - 仅进行简单的下采样操作
       - 保持了原始图像的边缘锐度
       - 但忽略了真实成像系统的物理限制
       - 在数学指标(如PSNR)上可能表现更好，但不一定符合真实情况

    2. 物理退化 (Physical Degradation):
       - 模拟真实成像过程: PSF模糊 + 下采样 + 噪声
       - PSF(点扩散函数)模拟了光学系统的衍射和像差
       - 下采样模拟了传感器像元合并
       - 噪声模拟了传感器噪声
       - 虽然指标可能稍差，但更接近真实低分辨率图像特性

    3. 在光谱融合反演中的应用:
       - 直接重采样: 适用于理想情况下的算法验证
       - 物理退化: 更适合构建真实的训练数据对
       - 选择哪种方法取决于应用场景和需求
    """)
    
    # 7. 保存结果
    print("\n" + "-"*50)
    print("保存结果到文件...")
    
    # 保存低分辨率图像
    output_dir = "degradation_results"
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存为PNG图像
    cv2.imwrite(f"{output_dir}/hr_band{band_to_display+1}.png", 
                (hr_img[band_to_display] * 255).astype(np.uint8))
    cv2.imwrite(f"{output_dir}/direct_resample_band{band_to_display+1}.png", 
                (resampled_img[band_to_display] * 255).astype(np.uint8))
    cv2.imwrite(f"{output_dir}/physical_degrade_band{band_to_display+1}.png", 
                (physical_img[band_to_display] * 255).astype(np.uint8))
    
    # 保存指标到文本文件
    with open(f"{output_dir}/metrics_comparison.txt", "w", encoding="utf-8") as f:
        f.write("直接重采样 vs 物理退化 质量指标对比\n")
        f.write("="*50 + "\n\n")
        f.write(f"图像: {uav_path}\n")
        f.write(f"下采样因子: {scale_factor}\n")
        f.write(f"PSF类型: {psf_type}\n")
        f.write(f"评估波段: {band_to_display+1}\n\n")
        
        f.write("直接重采样指标:\n")
        for key, value in resampled_metrics.items():
            if value is not None:
                f.write(f"  {key}: {value:.4f}\n")
        
        f.write("\n物理退化指标:\n")
        for key, value in physical_metrics.items():
            if value is not None:
                f.write(f"  {key}: {value:.4f}\n")
    
    print(f"结果已保存到 {output_dir} 目录")
    print("\n程序执行完成!")

if __name__ == "__main__":
    main()