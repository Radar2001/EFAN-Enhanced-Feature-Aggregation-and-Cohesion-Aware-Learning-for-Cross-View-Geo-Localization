import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import copy
from tqdm import tqdm
import time
import random
import math


def get_data(path):
    """获取DenseUAV数据 - 改进版，支持多层嵌套目录"""
    data = {}

    if not os.path.exists(path):
        print(f"警告: 路径不存在: {path}")
        return data

    # 查找所有包含图像的目录（假设目录名是ID）
    for root, dirs, files in os.walk(path):
        # 如果当前目录下有图像文件，将其作为ID
        image_files = []
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                image_files.append(file)

        if image_files:
            # 使用目录名作为ID（最后一级目录名）
            dir_name = os.path.basename(root)
            if dir_name not in data:
                data[dir_name] = {"path": root, "files": []}
            data[dir_name]["files"].extend(image_files)

    print(f"从 {path} 找到 {len(data)} 个ID，总计 {sum(len(info['files']) for info in data.values())} 个图像")
    return data


def parse_gps_coordinate(coord_str):
    """解析GPS坐标字符串"""
    if not coord_str:
        return 0.0

    # 处理带有方向标识的坐标
    if coord_str[0] in ['E', 'N']:
        # 东经、北纬为正
        return float(coord_str[1:])
    elif coord_str[0] in ['W', 'S']:
        # 西经、南纬为负
        return -float(coord_str[1:])
    else:
        # 尝试直接转换
        try:
            return float(coord_str)
        except:
            return 0.0


def load_gps_info(gps_file):
    """加载GPS信息 - 修正版"""
    gps_dict = {}
    try:
        with open(gps_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    # 使用完整路径作为键
                    img_path = parts[0]  # 如: train/satellite/000000/H80.tif

                    # 注意：根据你的GPS文件格式，可能需要调整索引
                    # 格式可能是: path longitude latitude height
                    # 或: path latitude longitude height
                    lon = parse_gps_coordinate(parts[1])  # 经度
                    lat = parse_gps_coordinate(parts[2])  # 纬度
                    height = float(parts[3])

                    gps_dict[img_path] = {
                        'latitude': lat,
                        'longitude': lon,
                        'height': height
                    }
        print(f"从 {gps_file} 加载了 {len(gps_dict)} 个GPS记录")
    except Exception as e:
        print(f"加载GPS信息失败: {e}")

    return gps_dict


class DenseUAVDatasetTrain(Dataset):
    """DenseUAV训练数据集"""

    def __init__(self,
                 data_root,
                 gps_train_file=None,
                 transforms_drone=None,
                 transforms_satellite=None,
                 prob_flip=0.5,
                 shuffle_batch_size=128):
        super().__init__()

        print("加载DenseUAV训练数据...")

        # 构建训练数据路径
        drone_folder = os.path.join(data_root, "train", "drone")
        satellite_folder = os.path.join(data_root, "train", "satellite")

        print(f"无人机训练数据路径: {drone_folder}")
        print(f"卫星训练数据路径: {satellite_folder}")

        if not os.path.exists(drone_folder):
            print(f"错误: 无人机训练数据文件夹不存在: {drone_folder}")
            drone_folder = os.path.join(data_root, "drone")
            print(f"尝试备用路径: {drone_folder}")

        if not os.path.exists(satellite_folder):
            print(f"错误: 卫星训练数据文件夹不存在: {satellite_folder}")
            satellite_folder = os.path.join(data_root, "satellite")
            print(f"尝试备用路径: {satellite_folder}")

        # 加载数据
        self.drone_dict = get_data(drone_folder)
        self.satellite_dict = get_data(satellite_folder)

        # 只使用同时存在于两个文件夹的ID
        self.ids = list(set(self.drone_dict.keys()).intersection(self.satellite_dict.keys()))
        self.ids.sort()

        print(f"找到 {len(self.ids)} 个训练ID")

        # 加载GPS信息
        self.gps_info = {}
        if gps_train_file and os.path.exists(gps_train_file):
            self.gps_info = load_gps_info(gps_train_file)
        else:
            # 尝试在数据根目录下查找GPS文件
            possible_gps_files = [
                os.path.join(data_root, "Dense_GPS_train.txt"),
                os.path.join(data_root, "train", "Dense_GPS_train.txt"),
                os.path.join(data_root, "Dense_GPS_ALL.txt")
            ]
            for gps_file in possible_gps_files:
                if os.path.exists(gps_file):
                    self.gps_info = load_gps_info(gps_file)
                    break

        # 创建训练对
        self.pairs = []

        for idx in self.ids:
            drone_path = self.drone_dict[idx]["path"]
            drone_imgs = self.drone_dict[idx]["files"]

            sat_path = self.satellite_dict[idx]["path"]
            sat_imgs = self.satellite_dict[idx]["files"]

            # 每个无人机图像与所有卫星图像配对
            for drone_img in drone_imgs:
                drone_img_path = os.path.join(drone_path, drone_img)
                drone_gps = self.gps_info.get(drone_img, {})

                for sat_img in sat_imgs:
                    sat_img_path = os.path.join(sat_path, sat_img)
                    sat_gps = self.gps_info.get(sat_img, {})

                    self.pairs.append((idx, drone_img_path, sat_img_path, drone_gps, sat_gps))

        print(f"创建了 {len(self.pairs)} 个训练对")

        self.transforms_drone = transforms_drone
        self.transforms_satellite = transforms_satellite
        self.prob_flip = prob_flip
        self.shuffle_batch_size = shuffle_batch_size

        self.samples = copy.deepcopy(self.pairs)

    def __getitem__(self, index):
        idx, drone_img_path, sat_img_path, drone_gps, sat_gps = self.samples[index]

        # 读取无人机图像
        drone_img = cv2.imread(drone_img_path)
        if drone_img is None:
            print(f"警告: 无法读取无人机图像: {drone_img_path}")
            drone_img = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            drone_img = cv2.cvtColor(drone_img, cv2.COLOR_BGR2RGB)

        # 读取卫星图像
        sat_img = cv2.imread(sat_img_path)
        if sat_img is None:
            print(f"警告: 无法读取卫星图像: {sat_img_path}")
            sat_img = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            sat_img = cv2.cvtColor(sat_img, cv2.COLOR_BGR2RGB)

        # 随机水平翻转
        if np.random.random() < self.prob_flip:
            drone_img = cv2.flip(drone_img, 1)
            sat_img = cv2.flip(sat_img, 1)

        # 图像变换
        if self.transforms_drone is not None:
            drone_img = self.transforms_drone(image=drone_img)['image']

        if self.transforms_satellite is not None:
            sat_img = self.transforms_satellite(image=sat_img)['image']

        # 转换为整数ID
        try:
            label = int(idx)
        except:
            label = hash(idx) % 1000000

        return drone_img, sat_img, label, drone_gps, sat_gps

    def __len__(self):
        return len(self.samples)

    def shuffle(self):
        """自定义shuffle函数"""
        print("\nShuffle Dataset:")

        pair_pool = copy.deepcopy(self.pairs)
        random.shuffle(pair_pool)

        pairs_epoch = set()
        idx_batch = set()
        batches = []
        current_batch = []

        break_counter = 0
        pbar = tqdm()

        while True:
            pbar.update()

            if len(pair_pool) > 0:
                pair = pair_pool.pop(0)
                idx, _, _, _, _ = pair

                if idx not in idx_batch and pair not in pairs_epoch:
                    idx_batch.add(idx)
                    current_batch.append(pair)
                    pairs_epoch.add(pair)
                    break_counter = 0
                else:
                    if pair not in pairs_epoch:
                        pair_pool.append(pair)
                    break_counter += 1

                if break_counter >= 512:
                    break
            else:
                break

            if len(current_batch) >= self.shuffle_batch_size:
                batches.extend(current_batch)
                idx_batch = set()
                current_batch = []

        pbar.close()
        time.sleep(0.3)

        self.samples = batches

        print(f"原始长度: {len(self.pairs)} - Shuffle后长度: {len(self.samples)}")
        print(f"Break Counter: {break_counter}")
        print(f"排除的对数: {len(self.pairs) - len(self.samples)}")
        if len(self.samples) > 0:
            print(f"第一个元素ID: {self.samples[0][0]} - 最后一个元素ID: {self.samples[-1][0]}")


class DenseUAVDatasetEval(Dataset):
    """DenseUAV评估数据集"""

    def __init__(self,
                 data_folder,
                 mode,  # 'query_drone' 或 'gallery_satellite'
                 gps_file=None,
                 transforms=None,
                 sample_ids=None):
        super().__init__()

        print(f"加载DenseUAV {mode} 数据...")

        if not os.path.exists(data_folder):
            print(f"错误: 数据文件夹不存在: {data_folder}")

        self.data_dict = get_data(data_folder)
        self.ids = list(self.data_dict.keys())
        self.ids.sort()

        print(f"找到 {len(self.ids)} 个{mode} ID")

        # 加载GPS信息
        self.gps_info = {}
        if gps_file and os.path.exists(gps_file):
            self.gps_info = load_gps_info(gps_file)
        else:
            # 尝试在上级目录查找GPS文件
            parent_dir = os.path.dirname(data_folder)
            possible_gps_files = [
                os.path.join(parent_dir, "Dense_GPS_test.txt"),
                os.path.join(parent_dir, "..", "Dense_GPS_test.txt"),
                os.path.join(parent_dir, "..", "Dense_GPS_ALL.txt")
            ]
            for gps_file_path in possible_gps_files:
                if os.path.exists(gps_file_path):
                    self.gps_info = load_gps_info(gps_file_path)
                    break

        self.mode = mode
        self.transforms = transforms
        self.given_sample_ids = sample_ids

        self.images = []
        self.sample_ids = []
        self.gps_data = []
        self.img_names = []  # 存储图像文件名

        # 收集图像和GPS信息
        for sample_id in self.ids:
            for file in self.data_dict[sample_id]["files"]:
                img_path = os.path.join(self.data_dict[sample_id]['path'], file)
                self.images.append(img_path)
                self.sample_ids.append(sample_id)
                self.img_names.append(file)

                # 获取GPS信息
                gps = self.gps_info.get(file, {})
                self.gps_data.append(gps)

        print(f"总共 {mode} 图像: {len(self.images)}")

        # 统计有GPS信息的图像数量
        has_gps_count = sum(1 for gps in self.gps_data if gps)
        print(f"{has_gps_count}个图像包含GPS信息")

    def __getitem__(self, index):
        img_path = self.images[index]
        sample_id = self.sample_ids[index]
        gps = self.gps_data[index]

        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告: 无法读取图像: {img_path}")
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 应用数据增强
        if self.transforms is not None:
            img = self.transforms(image=img)['image']

        # 转换为标签
        try:
            label = int(sample_id)
        except:
            label = hash(sample_id) % 1000000

        # 如果给定了sample_ids，检查是否在列表中
        if self.given_sample_ids is not None:
            if sample_id not in self.given_sample_ids:
                label = -1

        return img, label, gps

    def __len__(self):
        return len(self.images)

    def get_sample_ids(self):
        return set(self.sample_ids)

    def get_img_names(self):
        """获取图像文件名列表"""
        return self.img_names


def get_transforms(img_size,
                   mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]):
    """获取DenseUAV的transforms"""

    val_transforms = A.Compose([
        A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_AREA, p=1.0),
        A.Normalize(mean, std),
        ToTensorV2(),
    ])

    # 无人机图像的数据增强
    train_drone_transforms = A.Compose([
        A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_AREA, p=1.0),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.OneOf([
            A.Blur(blur_limit=3, p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
        ], p=0.3),
        # 修正的CoarseDropout参数
        A.CoarseDropout(num_holes=8, max_h_size=32, max_w_size=32, p=0.2),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean, std),
        ToTensorV2(),
    ])

    # 卫星图像的数据增强
    train_sat_transforms = A.Compose([
        A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_AREA, p=1.0),
        A.Affine(scale=(0.9, 1.1), translate_percent=(-0.1, 0.1),
                 rotate=(-10, 10), shear=(-5, 5), p=0.5),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05, p=0.5),
        # 修正的CoarseDropout参数
        A.CoarseDropout(num_holes=6, max_h_size=32, max_w_size=32, p=0.3),
        A.Normalize(mean, std),
        ToTensorV2(),
    ])

    return val_transforms, train_drone_transforms, train_sat_transforms


def print_directory_structure(path, indent=0):
    """打印目录结构，用于调试"""
    if not os.path.exists(path):
        print(" " * indent + f"[路径不存在]: {path}")
        return

    if os.path.isdir(path):
        print(" " * indent + f"[目录]: {os.path.basename(path)}/")
        try:
            items = os.listdir(path)
            for item in items[:10]:  # 只显示前10个
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path):
                    print_directory_structure(item_path, indent + 2)
                else:
                    print(" " * (indent + 2) + f"{item}")
            if len(items) > 10:
                print(" " * (indent + 2) + f"... 还有 {len(items) - 10} 个项")
        except PermissionError:
            print(" " * (indent + 2) + "[权限被拒绝]")
    else:
        print(" " * indent + f"[文件]: {os.path.basename(path)}")


# ==================== 测试代码 ====================

if __name__ == "__main__":
    # 示例用法
    data_root = "D:\BaiduNetdiskDownload\DenseUAV\DenseUAV"  # 替换为你的数据路径

    print("=" * 70)
    print("测试DenseUAV数据集加载")
    print("=" * 70)

    # 首先检查目录结构
    print("\n检查目录结构:")
    print_directory_structure(data_root)

    # 检查训练数据
    train_drone_path = os.path.join(data_root, "train", "drone")
    train_sat_path = os.path.join(data_root, "train", "satellite")

    print(f"\n训练数据路径检查:")
    print(f"  无人机训练数据: {train_drone_path} - 存在: {os.path.exists(train_drone_path)}")
    print(f"  卫星训练数据: {train_sat_path} - 存在: {os.path.exists(train_sat_path)}")

    # 检查测试数据
    query_drone_path = os.path.join(data_root, "test", "query_drone")
    gallery_sat_path = os.path.join(data_root, "test", "gallery_satellite")

    print(f"\n测试数据路径检查:")
    print(f"  查询无人机数据: {query_drone_path} - 存在: {os.path.exists(query_drone_path)}")
    print(f"  图库卫星数据: {gallery_sat_path} - 存在: {os.path.exists(gallery_sat_path)}")

    # 检查GPS文件
    gps_files = [
        os.path.join(data_root, "Dense_GPS_train.txt"),
        os.path.join(data_root, "Dense_GPS_test.txt"),
        os.path.join(data_root, "Dense_GPS_ALL.txt")
    ]

    print(f"\nGPS文件检查:")
    for gps_file in gps_files:
        exists = os.path.exists(gps_file)
        print(f"  {os.path.basename(gps_file)}: {gps_file} - 存在: {exists}")

    # 创建训练数据集
    print("\n" + "=" * 70)
    print("创建训练数据集")
    print("=" * 70)

    train_dataset = DenseUAVDatasetTrain(
        data_root=data_root,
        gps_train_file=os.path.join(data_root, "Dense_GPS_train.txt") if os.path.exists(
            os.path.join(data_root, "Dense_GPS_train.txt")) else None
    )

    print(f"\n训练数据集长度: {len(train_dataset)}")
    if len(train_dataset) > 0:
        # 显示一些样本信息
        print("\n训练数据集样本示例:")
        for i in range(min(3, len(train_dataset))):
            drone_img, sat_img, label, drone_gps, sat_gps = train_dataset[i]
            print(f"\n样本 {i}:")
            print(f"  ID: {label}")
            print(f"  无人机图像形状: {drone_img.shape}")
            print(f"  卫星图像形状: {sat_img.shape}")
            if drone_gps:
                print(f"  无人机GPS: 经度={drone_gps.get('longitude', 'N/A'):.6f}, "
                      f"纬度={drone_gps.get('latitude', 'N/A'):.6f}")
            if sat_gps:
                print(f"  卫星GPS: 经度={sat_gps.get('longitude', 'N/A'):.6f}, "
                      f"纬度={sat_gps.get('latitude', 'N/A'):.6f}")

    # 创建查询数据集（无人机查询）
    print("\n" + "=" * 70)
    print("创建查询数据集")
    print("=" * 70)

    query_drone_dataset = DenseUAVDatasetEval(
        data_folder=query_drone_path,
        mode='query_drone',
        gps_file=os.path.join(data_root, "Dense_GPS_test.txt") if os.path.exists(
            os.path.join(data_root, "Dense_GPS_test.txt")) else None
    )

    print(f"\n无人机查询数据集长度: {len(query_drone_dataset)}")

    # 创建图库数据集（卫星图库）
    gallery_sat_dataset = DenseUAVDatasetEval(
        data_folder=gallery_sat_path,
        mode='gallery_satellite',
        gps_file=os.path.join(data_root, "Dense_GPS_test.txt") if os.path.exists(
            os.path.join(data_root, "Dense_GPS_test.txt")) else None
    )

    print(f"卫星图库数据集长度: {len(gallery_sat_dataset)}")

    # 测试数据增强
    print("\n" + "=" * 70)
    print("测试数据增强")
    print("=" * 70)

    val_transforms, train_drone_transforms, train_sat_transforms = get_transforms((224, 224))

    if len(train_dataset) > 0:
        # 测试数据增强
        drone_img, sat_img, label, _, _ = train_dataset[0]

        print(
            f"原始无人机图像类型: {type(drone_img)}, 形状: {drone_img.shape if hasattr(drone_img, 'shape') else 'N/A'}")
        print(f"原始卫星图像类型: {type(sat_img)}, 形状: {sat_img.shape if hasattr(sat_img, 'shape') else 'N/A'}")

    print("\n数据集加载测试完成!")