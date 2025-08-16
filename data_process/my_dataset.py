import os                             # 导入 os 模块，用于文件和目录操作
from PIL import Image, ImageFile      # 从 Pillow 库中到入 Image\ImageFile 类，用于图像读取和处理
from torch.utils.data import Dataset  # 从 torch.utils.data 模块中导入 Dataset 类，用于自定义数据集

# 允许加载截断的图像，并增加最大图像像素限制
ImageFile.LOAD_TRUNCATED_IMAGES = True  # 防止加载损坏的图像时崩溃
Image.MAX_IMAGE_PIXELS = None  # 取消 Pillow 的默认像素限制

class MyDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root            # 数据集的根目录
        self.transform = transform  # 保存图像预处理函数
        self.samples = []           # 初始化一个列表，保存图像路径和对应标签

        for label in sorted(os.listdir(root)):      # 遍历根目录下的所有子文件夹，子文件夹名代表类别名称
            label_path = os.path.join(root, label)  # 获取子目录的完整路径
            if not os.path.isdir(label_path):       # 如果不是子目录（可能是文件），跳过
                continue
            for fname in os.listdir(label_path):    # 遍历子目录下的所有文件
                if fname.lower().endswith('.png'):  # 检查 .png（不区分大小写）
                    # 将 图像路径 和 对应标签（目录名转int） 添加到 samples 列表中
                    self.samples.append((os.path.join(label_path, fname), int(label)))
    
    def __len__(self):
        return len(self.samples)  # 返回样本数量
    
    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        
        try:
            # 尝试加载图像，并移除可能损坏的 ICC 配置文件
            image = Image.open(image_path)
            
            # 确保图像是 RGB 模式（如果不是，转换）
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 应用 transform（如果有）
            if self.transform:
                image = self.transform(image)
            
            return image, label
        
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # 如果出错，返回一个空图像或跳过（这里返回下一个样本）
            return self.__getitem__((idx + 1) % len(self))
    
