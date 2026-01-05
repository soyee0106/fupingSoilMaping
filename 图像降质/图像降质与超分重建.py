import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage, signal

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_images(images, titles, rows, cols, figsize=(15, 10)):
    """绘制多张图像"""
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.ravel() if rows * cols > 1 else [axes]
    
    for idx, (img, title) in enumerate(zip(images, titles)):
        axes[idx].imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        axes[idx].set_title(title, fontsize=12)
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()

# ====================== 1. 创建原始高分辨率图像 ======================
print("1. 创建原始高分辨率图像...")
# 创建一个简单的高分辨率图像 (200x200)，包含清晰的形状和纹理
hr_size = (200, 200)
hr_image = np.zeros(hr_size, dtype=np.float32)

# 添加一些清晰的形状：矩形、圆形和斜线
cv2.rectangle(hr_image, (20, 20), (80, 80), 0.8, -1)  # 白色矩形
cv2.circle(hr_image, (150, 60), 30, 0.6, -1)  # 灰色圆形
cv2.line(hr_image, (30, 150), (170, 150), 0.9, 3)  # 水平线
cv2.line(hr_image, (100, 30), (100, 170), 0.7, 3)  # 垂直线

# 添加一些高频纹理（棋盘格）
for i in range(0, 200, 20):
    for j in range(0, 200, 20):
        if (i//20 + j//20) % 2 == 0:
            hr_image[i:i+10, j:j+10] += 0.2

# 添加一些随机噪声模拟真实图像细节
hr_image += np.random.normal(0, 0.02, hr_size)
hr_image = np.clip(hr_image, 0, 1)

print(f"高分辨率图像尺寸: {hr_image.shape}")

# ====================== 2. 定义降质模型 ======================
print("\n2. 应用降质模型...")
print("降质过程: HR → 模糊 → 下采样 → 加噪 → LR")

def degradation_model(hr_img, scale_factor=4, blur_size=5, noise_level=0.05):
    """
    降质模型: 模拟图像质量下降的过程
    
    参数:
        hr_img: 高分辨率图像
        scale_factor: 下采样比例
        blur_size: 模糊核大小
        noise_level: 噪声水平
    
    返回:
        lr_img: 低分辨率图像
    """
    # 步骤1: 模糊 (模拟相机失焦或运动模糊)
    # 创建高斯模糊核
    blur_kernel = np.outer(
        signal.windows.gaussian(blur_size, blur_size/3),
        signal.windows.gaussian(blur_size, blur_size/3)
    )
    blur_kernel /= blur_kernel.sum()
    
    blurred = signal.convolve2d(hr_img, blur_kernel, mode='same', boundary='symm')
    
    # 步骤2: 下采样 (降低分辨率)
    # 使用OpenCV的resize进行下采样（平均池化）
    lr_height = hr_img.shape[0] // scale_factor
    lr_width = hr_img.shape[1] // scale_factor
    
    # 这里使用INTER_AREA插值，它适合下采样
    lr_img = cv2.resize(blurred, (lr_width, lr_height), interpolation=cv2.INTER_AREA)
    
    # 步骤3: 加噪 (模拟传感器噪声)
    noise = np.random.normal(0, noise_level, lr_img.shape)
    lr_img += noise
    
    # 确保像素值在合理范围
    lr_img = np.clip(lr_img, 0, 1)
    
    return lr_img, blur_kernel

# 应用降质模型
scale_factor = 4
lr_image, blur_kernel = degradation_model(hr_image, scale_factor=scale_factor)
print(f"低分辨率图像尺寸: {lr_image.shape}")
print(f"模糊核大小: {blur_kernel.shape}")

# ====================== 3. 超分重建方法对比 ======================
print("\n3. 尝试不同的超分重建方法...")

# 方法1: 最近邻插值 (最基础的方法)
def nearest_neighbor_upscale(img, scale_factor):
    """最近邻插值上采样"""
    h, w = img.shape
    new_h, new_w = h * scale_factor, w * scale_factor
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

# 方法2: 双线性插值
def bilinear_upscale(img, scale_factor):
    """双线性插值上采样"""
    h, w = img.shape
    new_h, new_w = h * scale_factor, w * scale_factor
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

# 方法3: 双三次插值 (传统方法中效果较好)
def bicubic_upscale(img, scale_factor):
    """双三次插值上采样"""
    h, w = img.shape
    new_h, new_w = h * scale_factor, w * scale_factor
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

# 方法4: 简单的反卷积重建 (模拟"逆向"降质过程)
def deconvolution_upscale(img, scale_factor, blur_kernel):
    """使用反卷积尝试恢复细节"""
    # 首先双三次上采样
    upsampled = bicubic_upscale(img, scale_factor)
    
    # 尝试使用维纳滤波进行反卷积 (简化版本)
    # 在实际中，我们会使用更复杂的正则化方法
    kernel_fft = np.fft.fft2(blur_kernel, s=upsampled.shape)
    image_fft = np.fft.fft2(upsampled)
    
    # 维纳滤波参数 (简化版)
    K = 0.01
    wiener_filter = np.conj(kernel_fft) / (np.abs(kernel_fft)**2 + K)
    
    deconvolved = np.fft.ifft2(image_fft * wiener_filter)
    deconvolved = np.abs(deconvolved)
    deconvolved = np.clip(deconvolved, 0, 1)
    
    return deconvolved

# 方法5: 简单的边缘增强方法 (模拟深度学习的效果)
def edge_enhanced_upscale(img, scale_factor):
    """结合边缘增强的超分方法"""
    # 首先双三次上采样
    upsampled = bicubic_upscale(img, scale_factor)
    
    # 使用Sobel算子检测边缘
    sobel_x = cv2.Sobel(upsampled, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(upsampled, cv2.CV_64F, 0, 1, ksize=3)
    edges = np.sqrt(sobel_x**2 + sobel_y**2)
    edges = edges / np.max(edges)  # 归一化
    
    # 增强边缘区域
    enhanced = upsampled + 0.3 * edges
    enhanced = np.clip(enhanced, 0, 1)
    
    return enhanced

# 应用不同的超分方法
print("应用不同重建方法...")
nn_result = nearest_neighbor_upscale(lr_image, scale_factor)
bilinear_result = bilinear_upscale(lr_image, scale_factor)
bicubic_result = bicubic_upscale(lr_image, scale_factor)
deconv_result = deconvolution_upscale(lr_image, scale_factor, blur_kernel)
edge_result = edge_enhanced_upscale(lr_image, scale_factor)

# ====================== 4. 评估重建质量 ======================
print("\n4. 评估重建质量...")

def calculate_psnr(img1, img2):
    """计算PSNR (峰值信噪比)"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_ssim(img1, img2):
    """计算SSIM (结构相似性) - 简化版本"""
    # 简化版SSIM计算，使用滑动窗口
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

# 计算各种方法的PSNR和SSIM
methods = {
    '最近邻': nn_result,
    '双线性': bilinear_result,
    '双三次': bicubic_result,
    '反卷积': deconv_result,
    '边缘增强': edge_result
}

print("质量评估 (PSNR越高越好, SSIM越接近1越好):")
print("-" * 50)
print(f"{'方法':<10} {'PSNR(dB)':<12} {'SSIM':<10}")
print("-" * 50)

for name, result in methods.items():
    # 确保尺寸匹配
    if result.shape != hr_image.shape:
        result = cv2.resize(result, (hr_image.shape[1], hr_image.shape[0]))
    
    psnr = calculate_psnr(hr_image, result)
    ssim = calculate_ssim(hr_image, result)
    print(f"{name:<10} {psnr:<12.2f} {ssim:<10.4f}")

# ====================== 5. 可视化结果 ======================
print("\n5. 可视化结果...")

# 准备可视化图像
images_to_show = [
    hr_image,                    # 原始HR
    lr_image,                    # 降质后的LR
    nn_result,                   # 最近邻重建
    bicubic_result,              # 双三次重建
    deconv_result,               # 反卷积重建
    edge_result                  # 边缘增强重建
]

titles = [
    '原始高分辨率图像 (HR)',
    f'降质后的低分辨率图像 (LR) {lr_image.shape[1]}x{lr_image.shape[0]}',
    f'最近邻插值重建\nPSNR: {calculate_psnr(hr_image, nn_result):.2f}dB',
    f'双三次插值重建\nPSNR: {calculate_psnr(hr_image, bicubic_result):.2f}dB',
    f'反卷积重建\nPSNR: {calculate_psnr(hr_image, deconv_result):.2f}dB',
    f'边缘增强重建\nPSNR: {calculate_psnr(hr_image, edge_result):.2f}dB'
]

# 绘制所有结果
plot_images(images_to_show, titles, rows=2, cols=3, figsize=(15, 10))

# ====================== 6. 细节对比 ======================
print("\n6. 细节区域对比...")

# 选择一个包含边缘和纹理的细节区域进行放大观察
crop_region = (slice(80, 130), slice(80, 130))  # 50x50的区域

detail_images = [
    hr_image[crop_region],
    cv2.resize(lr_image, (200, 200), interpolation=cv2.INTER_NEAREST)[crop_region],
    nn_result[crop_region],
    bicubic_result[crop_region],
    edge_result[crop_region]
]

detail_titles = [
    '原始HR细节',
    '直接放大LR细节',
    '最近邻重建细节',
    '双三次重建细节',
    '边缘增强重建细节'
]

fig, axes = plt.subplots(1, 5, figsize=(15, 4))
for idx, (img, title) in enumerate(zip(detail_images, detail_titles)):
    axes[idx].imshow(img, cmap='gray')
    axes[idx].set_title(title, fontsize=10)
    axes[idx].axis('off')
plt.tight_layout()
plt.show()

# ====================== 7. 降质过程可视化 ======================
print("\n7. 降质过程逐步可视化...")

# 逐步展示降质过程
hr_crop = hr_image[40:100, 40:100]  # 取一个60x60的区域

# 步骤1: 原始HR
# 步骤2: 模糊
blurred_crop = signal.convolve2d(hr_crop, blur_kernel, mode='same', boundary='symm')
# 步骤3: 下采样
downsampled_crop = cv2.resize(blurred_crop, (15, 15), interpolation=cv2.INTER_AREA)  # 60/4=15
# 步骤4: 加噪
noisy_crop = downsampled_crop + np.random.normal(0, 0.05, downsampled_crop.shape)
noisy_crop = np.clip(noisy_crop, 0, 1)
# 步骤5: 重建后
reconstructed_crop = bicubic_upscale(noisy_crop, 4)[:60, :60]  # 重新放大

degradation_steps = [
    hr_crop,
    blurred_crop,
    cv2.resize(downsampled_crop, (60, 60), interpolation=cv2.INTER_NEAREST),
    cv2.resize(noisy_crop, (60, 60), interpolation=cv2.INTER_NEAREST),
    reconstructed_crop
]

step_titles = [
    '1. 原始高分辨率图像',
    '2. 应用模糊 (高斯核)',
    '3. 下采样 (4倍缩小)',
    '4. 添加传感器噪声',
    '5. 重建后图像'
]

fig, axes = plt.subplots(1, 5, figsize=(15, 4))
for idx, (img, title) in enumerate(zip(degradation_steps, step_titles)):
    axes[idx].imshow(img, cmap='gray')
    axes[idx].set_title(title, fontsize=10)
    axes[idx].axis('off')
plt.tight_layout()
plt.show()

# ====================== 8. 总结与解释 ======================
print("\n" + "="*60)
print("总结与解释:")
print("="*60)
print("""
1. 降质过程 (Degradation Model):
   - 模糊: 使用高斯模糊核模拟相机失焦或运动模糊
   - 下采样: 将图像缩小到原来的1/4，丢失高频细节
   - 加噪: 添加高斯噪声模拟传感器噪声

2. 超分重建方法对比:
   - 最近邻插值: 最简单快速，但会产生块状伪影
   - 双线性插值: 中等质量，边缘较平滑但细节模糊
   - 双三次插值: 传统方法中效果较好，边缘更自然
   - 反卷积: 尝试逆向模糊过程，但容易放大噪声
   - 边缘增强: 增强高频细节，视觉上更清晰

3. 关键观察:
   - 所有方法都无法完全恢复原始细节（信息已永久丢失）
   - 反卷积方法理论上最合理（逆向降质过程），但对噪声敏感
   - 更高级的深度学习方法会学习图像先验，生成视觉合理的细节

4. 实际应用:
   - 传统插值方法：快速简单，适合实时应用
   - 深度学习方法：需要大量训练数据，但效果更好
   - 真实世界超分：需要估计真实降质模型，更具挑战性
""")