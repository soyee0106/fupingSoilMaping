"""
阶段一预训练模块

使用"卫星特征-真实无人机波段"数据对训练阶段一网络（传感器偏差解码器）。
损失函数：MSE（均方误差）
监控指标：R²（决定系数）
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional
import logging
from tqdm import tqdm
import numpy as np

from models.stage1_decoder import SensorBiasDecoder
from evaluation.metrics import calculate_r2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_stage1_decoder(
    model: SensorBiasDecoder,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    save_dir: Optional[Path] = None,
    save_best: bool = True,
    patience: int = 10
) -> Dict:
    """
    训练阶段一解码器（传感器偏差解码器）。
    
    参数:
        model: SensorBiasDecoder模型实例
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器（可选）
        num_epochs: 训练轮数
        learning_rate: 学习率
        device: 训练设备（'cuda'或'cpu'）
        save_dir: 模型保存目录
        save_best: 是否保存最佳模型
        patience: 早停耐心值（验证损失不下降的轮数）
    
    返回:
        包含训练历史的字典：
            - 'train_losses': 训练损失列表
            - 'val_losses': 验证损失列表（如果有验证集）
            - 'best_val_loss': 最佳验证损失
            - 'best_epoch': 最佳模型对应的轮数
    """
    logger.info("=" * 60)
    logger.info("Training Stage 1 Decoder (Sensor Bias Decoder)")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Number of epochs: {num_epochs}")
    logger.info(f"Learning rate: {learning_rate}")
    
    # 将模型移到设备
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练历史
    train_losses = []
    val_losses = []
    train_r2_scores = []
    val_r2_scores = []
    best_val_loss = float('inf')
    best_val_r2 = -float('inf')
    best_epoch = 0
    patience_counter = 0
    
    # 创建保存目录
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # 训练循环
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        # 用于计算R²的累积值
        all_preds = []
        all_targets = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in pbar:
            # 解包批次数据（字典格式）
            satellite_bands = batch['satellite_bands'].to(device)
            spectral_indices = batch['spectral_indices'].to(device)
            sensor_onehot = batch['sensor_onehot'].to(device)
            band_mask = batch['band_mask'].to(device)
            uav_bands_true = batch['uav_bands'].to(device)
            
            # 前向传播
            optimizer.zero_grad()
            uav_bands_pred = model(satellite_bands, spectral_indices, sensor_onehot, band_mask)
            
            # 计算损失
            loss = criterion(uav_bands_pred, uav_bands_true)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            # 累积预测值和真实值（用于计算R²）
            all_preds.append(uav_bands_pred.detach().cpu().numpy())
            all_targets.append(uav_bands_true.detach().cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # 计算训练集R²
        if all_preds:
            train_preds = np.concatenate(all_preds, axis=0)
            train_targets = np.concatenate(all_targets, axis=0)
            train_r2 = calculate_r2(train_targets.flatten(), train_preds.flatten())
            train_r2_scores.append(train_r2)
        else:
            train_r2 = 0.0
            train_r2_scores.append(train_r2)
        
        # 验证阶段
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            val_preds = []
            val_targets = []
            
            with torch.no_grad():
                for batch in val_loader:
                    satellite_bands = batch['satellite_bands'].to(device)
                    spectral_indices = batch['spectral_indices'].to(device)
                    sensor_onehot = batch['sensor_onehot'].to(device)
                    band_mask = batch['band_mask'].to(device)
                    uav_bands_true = batch['uav_bands'].to(device)
                    
                    uav_bands_pred = model(satellite_bands, spectral_indices, sensor_onehot, band_mask)
                    loss = criterion(uav_bands_pred, uav_bands_true)
                    
                    val_loss += loss.item()
                    val_batches += 1
                    
                    # 累积验证集预测值和真实值
                    val_preds.append(uav_bands_pred.cpu().numpy())
                    val_targets.append(uav_bands_true.cpu().numpy())
            
            avg_val_loss = val_loss / val_batches
            val_losses.append(avg_val_loss)
            
            # 计算验证集R²
            if val_preds:
                val_preds_flat = np.concatenate(val_preds, axis=0)
                val_targets_flat = np.concatenate(val_targets, axis=0)
                val_r2 = calculate_r2(val_targets_flat.flatten(), val_preds_flat.flatten())
                val_r2_scores.append(val_r2)
            else:
                val_r2 = 0.0
                val_r2_scores.append(val_r2)
            
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {avg_train_loss:.6f}, Train R²: {train_r2:.4f} | "
                f"Val Loss: {avg_val_loss:.6f}, Val R²: {val_r2:.4f}"
            )
            
            # 保存最佳模型（基于验证损失）
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_val_r2 = val_r2
                best_epoch = epoch + 1
                patience_counter = 0
                
                if save_best and save_dir is not None:
                    torch.save(
                        model.state_dict(),
                        save_dir / 'best_model.pth'
                    )
                    logger.info(f"  -> Best model saved (Val Loss: {best_val_loss:.6f}, Val R²: {best_val_r2:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        else:
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {avg_train_loss:.6f}, Train R²: {train_r2:.4f}"
            )
        
        # 保存每个epoch的模型（可选）
        if save_dir is not None and (epoch + 1) % 10 == 0:
            torch.save(
                model.state_dict(),
                save_dir / f'stage1_epoch_{epoch+1}.pth'
            )
    
    # 保存最终模型
    if save_dir is not None:
        torch.save(
            model.state_dict(),
            save_dir / 'stage1_final_model.pth'
        )
        logger.info(f"Final model saved to {save_dir / 'stage1_final_model.pth'}")
    
    logger.info("=" * 60)
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.6f} (epoch {best_epoch})")
    if val_r2_scores:
        logger.info(f"Best validation R²: {best_val_r2:.4f} (epoch {best_epoch})")
    logger.info("=" * 60)
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_r2_scores': train_r2_scores,
        'val_r2_scores': val_r2_scores,
        'best_val_loss': best_val_loss,
        'best_val_r2': best_val_r2 if val_r2_scores else None,
        'best_epoch': best_epoch
    }

