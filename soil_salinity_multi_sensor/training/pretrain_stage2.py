"""
阶段二预训练模块

使用"真实无人机波段及指数-盐分值"数据对训练阶段二网络（盐分反演网络）。
损失函数：MSE（均方误差）
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional
import logging
from tqdm import tqdm

from models.stage2_inverter import SalinityInverter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_stage2_inverter(
    model: SalinityInverter,
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
    训练阶段二反演网络（盐分反演网络）。
    
    参数:
        model: SalinityInverter模型实例
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器（可选）
        num_epochs: 训练轮数
        learning_rate: 学习率
        device: 训练设备（'cuda'或'cpu'）
        save_dir: 模型保存目录
        save_best: 是否保存最佳模型
        patience: 早停耐心值
    
    返回:
        包含训练历史的字典
    """
    logger.info("=" * 60)
    logger.info("Training Stage 2 Inverter (Salinity Inverter)")
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
    best_val_loss = float('inf')
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
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in pbar:
            # 解包批次数据
            uav_bands = batch[0].to(device)
            uav_indices = batch[1].to(device)
            salinity_true = batch[2].to(device)
            
            # 前向传播
            optimizer.zero_grad()
            salinity_pred = model(uav_bands, uav_indices)
            
            # 计算损失
            loss = criterion(salinity_pred, salinity_true)
            
            # 反向传播
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
                    uav_bands = batch[0].to(device)
                    uav_indices = batch[1].to(device)
                    salinity_true = batch[2].to(device)
                    
                    salinity_pred = model(uav_bands, uav_indices)
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
            
            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch + 1
                patience_counter = 0
                
                if save_best and save_dir is not None:
                    torch.save(
                        model.state_dict(),
                        save_dir / 'stage2_best_model.pth'
                    )
                    logger.info(f"  -> Best model saved (Val Loss: {best_val_loss:.6f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        else:
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.6f}")
        
        # 保存每个epoch的模型（可选）
        if save_dir is not None and (epoch + 1) % 10 == 0:
            torch.save(
                model.state_dict(),
                save_dir / f'stage2_epoch_{epoch+1}.pth'
            )
    
    # 保存最终模型
    if save_dir is not None:
        torch.save(
            model.state_dict(),
            save_dir / 'stage2_final_model.pth'
        )
        logger.info(f"Final model saved to {save_dir / 'stage2_final_model.pth'}")
    
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

