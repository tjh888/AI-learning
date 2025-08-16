# 训练配置参数
class Config:
    # 分布式设置
    MASTER_ADDR = 'localhost'
    MASTER_PORT = '12355'
    
    # 训练参数
    EPOCHS = 10            # 训练轮数
    TOTAL_BATCH_SIZE = 64  # 总的batch size
    BASE_LR = 0.001        # 初始学习率
    SEED = 2025            # 随机种子
    
    # 早停参数
    EARLY_STOP_PATIENCE = 3      # 早停轮数
    EARLY_STOP_MIN_DELTA = 0.01  # 早停的最小变化量
    
    # 数据路径
    TRAIN_DATA_PATH = 'data/train'  # 训练数据路径
    VAL_DATA_PATH = 'data/val'      # 验证数据路径
    
    # 输出路径
    OUTPUT_DIR = 'results'                                   # 输出目录
    MODEL_SAVE_PATH = 'results/best_model.pth'               # 模型保存路径
    PLOT_SAVE_PATH = 'results/loss_and_accuracy_curves.png'  # 绘图保存路径
    
    # 硬件相关
    NUM_WORKERS_PER_GPU = 4  # 每个GPU的worker数量