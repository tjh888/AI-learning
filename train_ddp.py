import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from tqdm import tqdm

# 从各模块导入
from config import Config
from utils.distributed import setup, cleanup
from utils.visualization import save_plots
from utils.early_stopping import EarlyStopper
from data_process.my_dataset import MyDataset
from data_process.transforms import transform
from models.ResNet import resnet34

def train(rank, world_size):
    # ---- 超参 ----
    cfg = Config()

    # ---- 初始化 ----
    setup(rank, world_size)
    torch.cuda.set_device(rank)                    # 设置当前使用的GPU

    if rank == 0:
        os.makedirs('results', exist_ok=True)      # 确保保存目录存在

    # ---- 数据增强 / 数据集 ----
    # 数据加载
    train_dataset = MyDataset(root=cfg.TRAIN_DATA_PATH, transform=transform)
    val_dataset = MyDataset(root=cfg.VAL_DATA_PATH, transform=transform)

    # 分布式采样器，使用 DistributedSampler 实现数据分片
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=cfg.SEED)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, seed=cfg.SEED)
    
    # ---- DataLoader 参数 ----
    batch_size = max(1, cfg.TOTAL_BATCH_SIZE // world_size)
    num_workers = max(0, cfg.NUM_WORKERS_PER_GPU * world_size)
    
    loader_args = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': torch.cuda.is_available(),
        'persistent_workers': num_workers > 0,  # 仅在 num_workers > 0 时启用
        'drop_last': True                       # 训练时通常 drop_last 保证同步 BN/梯度累积的一致性
    }
    # 只有在 num_workers>0 时设置 prefetch_factor（否则会报错）
    if num_workers > 0:
        loader_args['prefetch_factor'] = 4

    train_loader = DataLoader(train_dataset, sampler=train_sampler, **loader_args)
    val_loader = DataLoader(val_dataset, sampler=val_sampler, **{**loader_args, 'drop_last': False}, shuffle=False)
    
    # ---- 模型 & DDP ----
    model = resnet34().to(rank)
    
    # 只在主进程输出模型信息
    if rank == 0:
        print("\n" + "="*50)
        print(f"Using Model: {model.__class__.__name__}")
        print("="*50)
    
    model = DDP(model, device_ids=[rank])

    # 训练组件
    criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
    optimizer = optim.Adam(model.parameters(), lr=cfg.BASE_LR * world_size)  # 使用 Adam 优化器，基于 world_size 缩放 lr
    scaler = torch.amp.GradScaler()    # 混合精度训练
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS-3)  # 学习率调度器，warmup后使用cosine衰减
    
    # 训练指标记录（仅主进程需要）
    if rank == 0:
        train_losses, train_accuracies = [], []
        val_losses, val_accuracies = [], []
        early_stopper = EarlyStopper(patience=cfg.EARLY_STOP_PATIENCE,
                                     min_delta=cfg.EARLY_STOP_MIN_DELTA
                        )
        
    best_accuracy = 0.0
    best_model_path = 'results/best_model.pth'

    # 训练循环
    for epoch in range(cfg.EPOCHS):

        if epoch < 3:         # warmup：前 3 个 epoch
            lr = cfg.BASE_LR * (epoch + 1) / 3 * world_size
            for g in optimizer.param_groups:
                g['lr'] = lr
        else:
            scheduler.step()  # warmup后使用调度器
        
        dist.barrier()                  # 同步所有进程，确保同时进入新epoch
        train_sampler.set_epoch(epoch)  # 确保每个epoch数据顺序不同
        model.train()

        running_loss = torch.tensor(0.0).to(rank)  # 初始化当前 epoch 的损失
        correct_train = torch.tensor(0).to(rank)   # 当前 epoch 的正确预测样本数
        total_train = torch.tensor(0).to(rank)     # 当前 epoch 的样本总数

        # 仅在主进程显示进度
        disable_tqdm = (rank != 0)
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS} (rank {rank}) - Training", disable=disable_tqdm):
            # 将数据移动到 rank 上
            images = images.to(rank, non_blocking=True)
            labels = labels.to(rank, non_blocking=True)
            
            optimizer.zero_grad()                  # 梯度清零
            # 添加混合精度训练
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images)            # 前向传播
                loss = criterion(outputs, labels)  # 计算损失
            
            scaler.scale(loss).backward()          # 反向传播
            scaler.step(optimizer)                 # 更新参数
            scaler.update()

            # 计算训练集准确率
            bs = images.size(0)
            running_loss += loss.detach() * bs  # 累加损失，得到这个 epoch 的总损失
            total_train += bs                   # 累加样本数
            correct_train += (outputs.argmax(dim=1) == labels).sum().to(rank)  # 累加正确预测的样本数
            
        # 汇总所有进程的指标
        dist.all_reduce(running_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_train, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_train, op=dist.ReduceOp.SUM)

        # 主进程收集训练指标
        if rank == 0:
            train_loss = running_loss.item() / total_train.item()        # 计算当前 epoch 的训练损失
            train_accuracy = correct_train.item() / total_train.item()   # 计算当前 epoch 的训练集准确率
            train_losses.append(train_loss)                              # 记录每个 epoch 的平均损失（每 batch）
            train_accuracies.append(train_accuracy)                      # 记录每个 epoch 的训练集准确率
            print(f"Epoch {epoch+1}/{cfg.EPOCHS}, Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2%}")
        
        # 验证阶段
        model.eval()  # 设置模型为测试模式
        
        val_loss = torch.tensor(0.0, device=rank)   # 
        correct_val = torch.tensor(0 ,device=rank)  # 当前 epoch 的正确预测样本数
        total_val = torch.tensor(0 ,device=rank)    # 当前 epoch 的样本总数
        
        # 固定验证集顺序
        val_sampler.set_epoch(epoch)
        
        with torch.no_grad():                # 不计算梯度，节省内存
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS} (rank {rank}) - Validating", disable=disable_tqdm):

                images = images.to(rank, non_blocking=True)
                labels = labels.to(rank, non_blocking=True)

                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(images)  # 前向传播
                    loss = criterion(outputs, labels)
                
                val_loss += loss.detach() * images.size(0)
                total_val += images.size(0)  # 累加样本数
                correct_val += (outputs.argmax(dim=1) == labels).sum().to(rank)  # 累加正确预测的样本数
        
        dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_val, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_val, op=dist.ReduceOp.SUM)

        dist.barrier()  # 验证结束同步（确保所有进程完成验证后再继续）
        
        # 在验证结束后立即清理显存（减少显存占用峰值）
        torch.cuda.empty_cache()

        # ------ 验证结果 & 早停 ------
        # 所有进程先聚合验证结果
        val_loss = val_loss.item() / total_val.item()
        val_accuracy = correct_val.item() / total_val.item()

        if rank == 0:
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            print(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2%}")

            # 保存最佳模型
            if val_accuracy > best_accuracy :
                best_accuracy = val_accuracy
                checkpoint = {
                    'epoch': epoch,
                    'state_dict': model.module.state_dict(),  # 使用module获取原始模型
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),            # 保存scaler状态
                    'accuracy': best_accuracy,
                    'model': model.module.__class__.__name__,
                    'batch_size': cfg.TOTAL_BATCH_SIZE,
                    'base_lr': cfg.BASE_LR,
                    'world_size': world_size
                }
                torch.save(checkpoint, best_model_path)
                print(f"Best model saved with accuracy: {best_accuracy:.2%}")
            
            # 早停判定（仅在 rank 0 判定，再广播给所有进程）
            stop_training = early_stopper.should_stop(val_loss)
        else:
            stop_training = False
        
        # 将停止决定转换为tensor并广播
        stop_flag = torch.tensor(int(stop_training), device=rank)
        dist.broadcast(stop_flag, src=0)

        # 所有进程同步清理显存
        torch.cuda.empty_cache()

        # 统一退出训练循环
        if stop_flag.item() == 1:
            if rank == 0:
                print(f"Early stopping triggered at epoch {epoch+1}!")
            break
        
    # 训练结束后，主进程保存曲线
    if rank == 0:
        save_plots(train_losses, train_accuracies, val_losses, val_accuracies)
        # 输出数据统计
        total_train_samples = len(train_dataset)
        total_val_samples = len(val_dataset)
        print(f"\nTraining completed!")
        print(f"Total training samples: {total_train_samples}")
        print(f"Total validation samples: {total_val_samples}")
        print(f"Best validation accuracy: {best_accuracy:.2%}")
    
    torch.cuda.synchronize()      # 确保所有GPU操作完成    
    cleanup()                     # 清理分布式环境

if __name__ == "__main__":
    world_size = 2  # 使用2张GPU（2 * RTX 4090）
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size, join=True)
