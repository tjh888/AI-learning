import os  # 导入操作系统接口模块 - 用于文件和目录操作
import random  # 导入随机数模块 - 用于数据集的随机打乱
from functools import partial  # 从functools导入partial函数 - 用于创建偏函数
import multiprocessing  # 导入多进程处理模块 - 用于并行处理切片
from tqdm import tqdm  # 导入进度条模块 - 显示处理进度
import openslide  # 导入openslide库 - 用于读取.svs病理切片文件
import numpy as np  # 导入数值计算库 - 用于数组操作和数学计算
from PIL import Image  # 导入图像处理库 - 用于图像格式转换和保存
from collections import defaultdict  # 导入带默认值的字典 - 用于统计信息收集

SPLIT_RATIO = (0.7, 0.15, 0.15)  # 训练集70%，验证集15%，测试集15%

def is_background(tile, threshold=0.5):
    """
    判断图像块是否主要为背景
    :param tile: PIL.Image对象 - 输入图像块
    :param threshold: float - 背景判定阈值(0-1)
    :return: bool - 是否为背景
    """
    # 将图像转换为numpy数组
    arr = np.array(tile)  
    # 创建掩码：所有RGB通道值都大于220的像素为True
    white_mask = np.all(arr > 220, axis=2)  
    # 计算白色像素所占比例
    white_ratio = np.mean(white_mask)  
    # 返回是否超过阈值
    return white_ratio > threshold  

def check_slide_exists(full_slide_name, output_root, slide_type):
    """
    检查文件夹中是否已存在相同full_slide_name的文件
    :param full_slide_name: str - 完整切片名称
    :param output_root: str - 输出根目录
    :param slide_type: str - 切片类型（如normal, LUSC, LUAD）
    :return: bool - 是否已存在
    """
    # 遍历所有数据集分割类型
    for split in ["train", "val", "test"]:
        # 构建normal类型切片的目录路径
        slide_dir = os.path.join(output_root, split, slide_type)  
        # 检查目录是否存在
        if os.path.exists(slide_dir):  
            # 遍历目录中的文件
            for file in os.listdir(slide_dir):  
                # 检查是否有以full_slide_name开头的文件
                if file.startswith(full_slide_name):  # 用full_slide_name判断
                    return True  
    return False  

def process_slide(slide_path, output_root, split_choice, tile_size=512, level=1):
    """
    处理单个.svs文件
    :param slide_path: str - .svs文件路径
    :param output_root: str - 输出根目录
    :param split_choice: str - 数据集分割类型(train/val/test)
    :param tile_size: int - 切块大小(默认512)
    :param level: int - 金字塔层级(默认1)
    :return: tuple - (切片名称, 分割类型, 保存的切块数)
    """
    # 获取不带扩展名的文件名
    full_slide_name = os.path.splitext(os.path.basename(slide_path))[0]  
    # 按'-'分割文件名
    slide_name_parts = full_slide_name.split('-')  
    # 取前4部分作为切片名称
    slide_name = '-'.join(slide_name_parts[:4])  
    
    # 从文件名中提取类型代码
    slide_type_code = slide_name_parts[3][:2]  
    # 检查是否为数字
    if slide_type_code.isdigit():  
        slide_type_code = int(slide_type_code)  
        # 判断切片类型(1-9为LUAD，10-19为normal)
        if 1 <= slide_type_code <= 9:  
            slide_type = "LUAD"  
        elif 10 <= slide_type_code <= 19:  
            slide_type = "normal"  
        else:  
            slide_type = "unknown"
            # 所有合法类型均执行去重检查
        if slide_type in ["normal", "LUAD"]:
            if check_slide_exists(full_slide_name, output_root, slide_type):
                print(f"[SKIP] {full_slide_name} ({slide_type}) already exists, skipping...")
                return (full_slide_name, split_choice, 0)
    else:  
        slide_type = "unknown"  
    
    # 创建输出目录(按分割类型和切片类型)
    output_dir = os.path.join(output_root, split_choice, slide_type)  
    os.makedirs(output_dir, exist_ok=True)  

    try:
        # 打开.svs文件
        slide = openslide.OpenSlide(slide_path)  
    except Exception as e:
        print(f"[ERROR] Cannot open {slide_path}: {e}")  
        return (full_slide_name, split_choice, 0)  

    # 检查请求的层级是否有效
    if level >= slide.level_count:  
        level = slide.level_count - 1  
        print(f"[Warning] {full_slide_name} using level {level}")  

    # 获取选定层级的尺寸
    width, height = slide.level_dimensions[level]  
    # 计算x和y方向的切块数量
    x_tiles = width // tile_size  
    y_tiles = height // tile_size  

    total_saved = 0  # 保存的切块计数器

    # 遍历所有切块位置
    for i in range(x_tiles):  
        for j in range(y_tiles):  
            # 计算当前切块的起始坐标
            x = i * tile_size  
            y = j * tile_size  
            # 读取切块区域并转换为RGB
            tile = slide.read_region(  
                (int(x * slide.level_downsamples[level]), int(y * slide.level_downsamples[level])),  
                level, (tile_size, tile_size)  
            ).convert('RGB')  
            
            # 如果不是背景则保存
            if not is_background(tile):  
                total_saved += 1  
                # 生成切块文件名
                tile_filename = f"{full_slide_name}_{total_saved}.png"  
                # 保存切块图像
                tile.save(os.path.join(output_dir, tile_filename))  

    print(f"[DONE] {full_slide_name} ({slide_type} → {split_choice}): saved {total_saved} tiles")  
    return (full_slide_name, split_choice, total_saved)  

def get_all_slide_paths(input_root):
    """
    获取所有.svs文件路径
    :param input_root: str - 输入根目录
    :return: list - 所有.svs文件路径列表
    """
    svs_paths = []  
    # 遍历输入目录下的所有子文件夹
    for subfolder in os.listdir(input_root):  
        folder_path = os.path.join(input_root, subfolder)  
        # 跳过非目录项
        if not os.path.isdir(folder_path):  
            continue  
        # 遍历子文件夹中的文件
        for file in os.listdir(folder_path):  
            # 收集.svs文件
            if file.endswith(".svs"):  
                svs_paths.append(os.path.join(folder_path, file))  
    return svs_paths  

def process_slide_wrapper(args, output_root, tile_size, level):
    """
    包装函数用于多进程处理
    :param args: tuple - (slide_path, split_choice)
    :param output_root: str - 输出根目录
    :param tile_size: int - 切块大小
    :param level: int - 金字塔层级
    :return: tuple - process_slide的返回值
    """
    path, split = args  
    return process_slide(path, output_root, split, tile_size, level)  

def process_all_slides_parallel(input_root, output_root, tile_size=512, level=1, num_workers=None):
    """
    主处理函数 - 并行处理所有切片
    :param input_root: str - 输入根目录
    :param output_root: str - 输出根目录
    :param tile_size: int - 切块大小(默认512)
    :param level: int - 金字塔层级(默认1)
    :param num_workers: int - 进程数(默认None表示自动计算)
    """
    # 创建所有需要的输出目录
    for split in ["train", "val", "test"]:  
        for slide_type in ["LUSC", "normal", "LUAD"]:  
            os.makedirs(os.path.join(output_root, split, slide_type), exist_ok=True)  

    # 获取所有.svs文件路径
    slide_paths = get_all_slide_paths(input_root)  
    print(f"[INFO] Found {len(slide_paths)} .svs files.")  

    # 检查是否找到文件
    if not slide_paths:  
        print("[ERROR] No .svs files found.")  
        return  

    # 随机打乱文件顺序
    random.shuffle(slide_paths)  
    total = len(slide_paths)  
    # 计算各数据集分割点
    train_end = int(total * SPLIT_RATIO[0])  
    val_end = train_end + int(total * SPLIT_RATIO[1])  
    
    # 分配每个切片到不同的数据集分割
    slide_assignments = []  
    for i, path in enumerate(slide_paths):  
        if i < train_end:  
            split = "train"  
        elif i < val_end:  
            split = "val"  
        else:  
            split = "test"  
        slide_assignments.append((path, split))  

    # 设置进程数(默认为CPU核心数-1)
    if num_workers is None:  
        num_workers = max(1, multiprocessing.cpu_count() - 1)  

    print(f"[INFO] Using {num_workers} parallel processes...")  
    print(f"[SPLIT] Train: {train_end}, Val: {val_end-train_end}, Test: {total-val_end}")  

    # 初始化统计字典
    stats = defaultdict(lambda: defaultdict(int))  

    # 创建进程池
    with multiprocessing.Pool(num_workers) as pool:  
        # 创建偏函数，固定部分参数
        worker = partial(process_slide_wrapper,  
                        output_root=output_root,  
                        tile_size=tile_size,  
                        level=level)  
        
        # 使用进度条显示处理进度
        with tqdm(total=len(slide_assignments), desc="Processing slides") as pbar:  
            # 并行处理所有切片
            for result in pool.imap_unordered(worker, slide_assignments):  
                full_slide_name, split, count = result  
                # 更新统计信息
                if count > 0:  
                    stats[full_slide_name][split] = count  
                # 更新进度条
                pbar.update(1)  

    # 打印汇总统计
    print("\n=== Summary Statistics ===")  
    total_train = sum(v.get("train", 0) for v in stats.values())  
    total_val = sum(v.get("val", 0) for v in stats.values())  
    total_test = sum(v.get("test", 0) for v in stats.values())  
    print(f"Total tiles -> train: {total_train}, val: {total_val}, test: {total_test}")  

if __name__ == "__main__":
    # 输入目录 - 包含.svs文件的根目录
    input_root = r"F:\TCGA-LUAD"  
    # 输出目录 - 处理结果保存位置
    output_root = r"G:\LUAD_LUSC\data"  
    # 启动并行处理
    process_all_slides_parallel(input_root, output_root, tile_size=512, level=1)

