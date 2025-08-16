import os
import torch
import torch.distributed as dist

def setup(rank, world_size, seed=2025):
    """初始化分布式训练环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", init_method="env://", rank=rank, world_size=world_size)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    
def cleanup():
    """清理分布式环境"""
    dist.destroy_process_group()