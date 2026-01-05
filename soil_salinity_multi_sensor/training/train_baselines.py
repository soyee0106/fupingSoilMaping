"""
基线模型训练模块

训练Model A和Model B作为对比基线。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional
import logging
from tqdm import tqdm

from models.baseline_models import BaselineModelA, BaselineModelB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_baseline_a(
    model: BaselineModelA,
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
    训练基线模型A（直接从卫星特征预测盐分）。
    
    参数:
        model: BaselineModelA模型实例
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器（可选）
        num_epochs: 训练轮数
        learning_rate: 学习率
        device: 训练设备
        save_dir: 模型保存目录
        save_best: 是否保存最佳模型
        patience: 早停耐心值
    
    返回:
        包含训练历史的字典
    """
    logger.info("=" * 60)
    logger.info("Training Baseline Model A")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Number of epochs: {num_epochs}")
    logger.info(f"Learning rate: {learning_rate}")
    
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in pbar:
            satellite_bands = batch[0].to(device)
            spectral_indices = batch[1].to(device)
            sensor_onehot = batch[2].to(device)
            salinity_true = batch[3].to(device)
            
            optimizer.zero_grad()
            salinity_pred = model(satellite_bands, spectral_indices, sensor_onehot)
            loss = criterion(salinity_pred, salinity_true)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    satellite_bands = batch[0].to(device)
                    spectral_indices = batch[1].to(device)
                    sensor_onehot = batch[2].to(device)
                    salinity_true = batch[3].to(device)
                    
                    salinity_pred = model(satellite_bands, spectral_indices, sensor_onehot)
                    loss = criterion(salinity_pred, salinity_true)
                    
                    val_loss += loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches
            val_losses.append(avg_val_loss)
            
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {avg_train_loss:.6f}, "
                f"Val Loss: {avg_val_loss:.6f}"
            )
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch + 1
                patience_counter = 0
                
                if save_best and save_dir is not None:
                    torch.save(
                        model.state_dict(),
                        save_dir / 'baseline_a_best.pth'
                    )
                    logger.info(f"  -> Best model saved (Val Loss: {best_val_loss:.6f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        else:
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.6f}")
    
    if save_dir is not None:
        torch.save(
            model.state_dict(),
            save_dir / 'baseline_a_final.pth'
        )
    
    logger.info("=" * 60)
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.6f} (epoch {best_epoch})")
    logger.info("=" * 60)
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch
    }


def train_baseline_b(
    model: BaselineModelB,
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
    训练基线模型B（卫星->无人机波段->盐分，但不学习传感器偏差）。
    
    参数:
        model: BaselineModelB模型实例
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器（可选）
        num_epochs: 训练轮数
        learning_rate: 学习率
        device: 训练设备
        save_dir: 模型保存目录
        save_best: 是否保存最佳模型
        patience: 早停耐心值
    
    返回:
        包含训练历史的字典
    """
    logger.info("=" * 60)
    logger.info("Training Baseline Model B")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Number of epochs: {num_epochs}")
    logger.info(f"Learning rate: {learning_rate}")
    
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in pbar:
            satellite_bands = batch[0].to(device)
            satellite_indices = batch[1].to(device)
            sensor_onehot = batch[2].to(device)  # 不使用，但保留接口
            salinity_true = batch[3].to(device)
            
            optimizer.zero_grad()
            outputs = model(satellite_bands, satellite_indices, sensor_onehot)
            salinity_pred = outputs['salinity']
            loss = criterion(salinity_pred, salinity_true)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    satellite_bands = batch[0].to(device)
                    satellite_indices = batch[1].to(device)
                    sensor_onehot = batch[2].to(device)
                    salinity_true = batch[3].to(device)
                    
                    outputs = model(satellite_bands, satellite_indices, sensor_onehot)
                    salinity_pred = outputs['salinity']
                    loss = criterion(salinity_pred, salinity_true)
                    
                    val_loss += loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches
            val_losses.append(avg_val_loss)
            
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {avg_train_loss:.6f}, "
                f"Val Loss: {avg_val_loss:.6f}"
            )
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch + 1
                patience_counter = 0
                
                if save_best and save_dir is not None:
                    torch.save(
                        model.state_dict(),
                        save_dir / 'baseline_b_best.pth'
                    )
                    logger.info(f"  -> Best model saved (Val Loss: {best_val_loss:.6f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        else:
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.6f}")
    
    if save_dir is not None:
        torch.save(
            model.state_dict(),
            save_dir / 'baseline_b_final.pth'
        )
    
    logger.info("=" * 60)
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.6f} (epoch {best_epoch})")
    logger.info("=" * 60)
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch
    }

