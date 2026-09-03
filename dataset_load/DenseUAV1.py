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


def get_data(path):
    """获取DenseUAV数据"""
    data = {}

    if not os.path.exists(path):
        print(f"警告: 路径不存在: {path}")
        return data

    # 查找所有包含图像的目录
    for root, dirs, files in os.walk(path):
        image_files = []
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.JPG', '.TIF')):
                image_files.append(file)

        if image_files:
            dir_name = os.path.basename(root)
            if dir_name not in data:
                data[dir_name] = {"path": root, "files": []}
            data[dir_name]["files"].extend(image_files)

    return data


class DenseUAVDatasetTrain(Dataset):
    """DenseUAV训练数据集"""

    def __init__(self,
                 data_root,
                 transforms_drone=None,
                 transforms_satellite=None,
                 prob_flip=0.5,
                 shuffle_batch_size=128):
        super().__init__()

        drone_folder = os.path.join(data_root, "train", "drone")
        satellite_folder = os.path.join(data_root, "train", "satellite")

        self.drone_dict = get_data(drone_folder)
        self.satellite_dict = get_data(satellite_folder)

        self.ids = list(set(self.drone_dict.keys()).intersection(self.satellite_dict.keys()))
        self.ids.sort()

        print(f"找到 {len(self.ids)} 个训练ID")

        self.pairs = []

        for idx in self.ids:
            drone_path = self.drone_dict[idx]["path"]
            drone_imgs = self.drone_dict[idx]["files"]

            sat_path = self.satellite_dict[idx]["path"]
            sat_imgs = self.satellite_dict[idx]["files"]

            for drone_img in drone_imgs:
                drone_img_path = os.path.join(drone_path, drone_img)

                for sat_img in sat_imgs:
                    sat_img_path = os.path.join(sat_path, sat_img)

                    # 只保存前三个元素（ID，无人机路径，卫星路径）
                    self.pairs.append((idx, drone_img_path, sat_img_path))

        print(f"创建了 {len(self.pairs)} 个训练对")

        self.transforms_drone = transforms_drone
        self.transforms_satellite = transforms_satellite
        self.prob_flip = prob_flip
        self.shuffle_batch_size = shuffle_batch_size

        self.samples = copy.deepcopy(self.pairs)

    def __getitem__(self, index):
        idx, drone_img_path, sat_img_path = self.samples[index]

        # 读取无人机图像
        drone_img = cv2.imread(drone_img_path)
        if drone_img is None:
            drone_img = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            drone_img = cv2.cvtColor(drone_img, cv2.COLOR_BGR2RGB)

        # 读取卫星图像
        sat_img = cv2.imread(sat_img_path)
        if sat_img is None:
            sat_img = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            sat_img = cv2.cvtColor(sat_img, cv2.COLOR_BGR2RGB)

        # 随机水平翻转
        if np.random.random() < self.prob_flip:
            drone_img = cv2.flip(drone_img, 1)
            sat_img = cv2.flip(sat_img, 1)

        # 应用数据增强
        if self.transforms_drone is not None:
            drone_img = self.transforms_drone(image=drone_img)['image']

        if self.transforms_satellite is not None:
            sat_img = self.transforms_satellite(image=sat_img)['image']

        # 转换为整数ID
        try:
            label = int(idx)
        except:
            label = hash(idx) % 1000000

        return drone_img, sat_img, label

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
                idx = pair[0]  # 只获取ID
                pair_key = pair  # pair现在只有3个元素，都是可哈希的

                if idx not in idx_batch and pair_key not in pairs_epoch:
                    idx_batch.add(idx)
                    current_batch.append(pair)
                    pairs_epoch.add(pair_key)
                    break_counter = 0
                else:
                    if pair_key not in pairs_epoch:
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
                 mode,  # 'query' 或 'gallery_drone'
                 transforms=None,
                 sample_ids=None):
        super().__init__()

        print(f"加载DenseUAV {mode} 数据...")

        self.data_dict = get_data(data_folder)
        self.ids = list(self.data_dict.keys())
        self.ids.sort()

        print(f"找到 {len(self.ids)} 个{mode} ID")

        self.mode = mode
        self.transforms = transforms
        self.given_sample_ids = sample_ids

        self.images = []
        self.sample_ids = []

        for sample_id in self.ids:
            for file in self.data_dict[sample_id]["files"]:
                img_path = os.path.join(self.data_dict[sample_id]['path'], file)
                self.images.append(img_path)
                self.sample_ids.append(sample_id)

        print(f"总共 {mode} 图像: {len(self.images)}")

    def __getitem__(self, index):
        img_path = self.images[index]
        sample_id = self.sample_ids[index]

        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            img = self.transforms(image=img)['image']

        try:
            label = int(sample_id)
        except:
            label = hash(sample_id) % 1000000

        if self.given_sample_ids is not None:
            if sample_id not in self.given_sample_ids:
                label = -1

        return img, label

    def __len__(self):
        return len(self.images)

    def get_sample_ids(self):
        return set(self.sample_ids)


def get_transforms(img_size,
                   mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]):
    """获取DenseUAV的transforms"""

    val_transforms = A.Compose([
        A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_AREA, p=1.0),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
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
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ToTensorV2(),
    ])

    # 卫星图像的数据增强
    train_sat_transforms = A.Compose([
        A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_AREA, p=1.0),
        A.Affine(scale=(0.9, 1.1), translate_percent=(-0.1, 0.1),
                 rotate=(-10, 10), shear=(-5, 5), p=0.5),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05, p=0.5),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ToTensorV2(),
    ])

    return val_transforms, train_drone_transforms, train_sat_transforms