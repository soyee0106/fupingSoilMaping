"""
联合微调模块

将预训练好的阶段一和阶段二网络连接，使用"卫星特征-盐分值"数据
进行端到端微调。

关键点：
- 阶段一参数使用较低的学习率（如1e-5）
- 阶段二参数使用较高的学习率（如1e-4）
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional
import logging
from tqdm import tqdm

from models.full_model import FullModelC

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fine_tune_full_model(
    model: FullModelC,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    num_epochs: int = 50,
    stage1_lr: float = 1e-5,
    stage2_lr: float = 1e-4,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    save_dir: Optional[Path] = None,
    save_best: bool = True,
    patience: int = 10
) -> Dict:
    """
    端到端微调完整模型（Model C）。
    
    参数:
        model: FullModelC模型实例（应已加载预训练权重）
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器（可选）
        num_epochs: 训练轮数
        stage1_lr: 阶段一网络的学习率（较低）
        stage2_lr: 阶段二网络的学习率（较高）
        device: 训练设备（'cuda'或'cpu'）
        save_dir: 模型保存目录
        save_best: 是否保存最佳模型
        patience: 早停耐心值
    
    返回:
        包含训练历史的字典
    """
    logger.info("=" * 60)
    logger.info("Fine-tuning Full Model (Model C)")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Number of epochs: {num_epochs}")
    logger.info(f"Stage 1 learning rate: {stage1_lr}")
    logger.info(f"Stage 2 learning rate: {stage2_lr}")
    
    # 将模型移到设备
    model = model.to(device)
    
    # 定义损失函数
    criterion = nn.MSELoss()
    
    # 为不同阶段设置不同的学习率
    optimizer = optim.Adam([
        {'params': model.stage1.parameters(), 'lr': stage1_lr},
        {'params': model.stage2.parameters(), 'lr': stage2_lr}
    ])
    
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
            satellite_bands = batch[0].to(device)
            satellite_indices = batch[1].to(device)
            sensor_onehot = batch[2].to(device)
            salinity_true = batch[3].to(device)
            # 掩码向量（如果提供）
            band_mask = batch[4].to(device) if len(batch) > 4 else None
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(satellite_bands, satellite_indices, sensor_onehot, band_mask)
            salinity_pred = outputs['salinity']
            
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
                    satellite_bands = batch[0].to(device)
                    satellite_indices = batch[1].to(device)
                    sensor_onehot = batch[2].to(device)
                    salinity_true = batch[3].to(device)
                    # 掩码向量（如果提供）
                    band_mask = batch[4].to(device) if len(batch) > 4 else None
                    
                    outputs = model(satellite_bands, satellite_indices, sensor_onehot, band_mask)
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
            
            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch + 1
                patience_counter = 0
                
                if save_best and save_dir is not None:
                    torch.save(
                        model.state_dict(),
                        save_dir / 'full_model_best.pth'
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
                save_dir / f'full_model_epoch_{epoch+1}.pth'
            )
    
    # 保存最终模型
    if save_dir is not None:
        torch.save(
            model.state_dict(),
            save_dir / 'full_model_final.pth'
        )
        logger.info(f"Final model saved to {save_dir / 'full_model_final.pth'}")
    
    logger.info("=" * 60)
    logger.info("Fine-tuning completed!")
    logger.info(f"Best validation loss: {best_val_loss:.6f} (epoch {best_epoch})")
    logger.info("=" * 60)
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch
    }

