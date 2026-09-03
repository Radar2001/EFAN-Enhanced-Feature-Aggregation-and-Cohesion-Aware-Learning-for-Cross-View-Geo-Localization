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

import os
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
cv2.setLogLevel(0)  # 设置OpenCV日志级别为0（不输出任何日志）

def safe_imread(file_path):
    """安全的图像读取函数，使用备用方法"""
    try:
        # 方法1: 直接使用OpenCV
        img = cv2.imread(file_path)
        if img is not None:
            return img

        # 方法2: 使用备用方法（imdecode）
        with open(file_path, 'rb') as f:
            img_data = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        if img is not None:
            return img

        # 方法3: 如果前两种方法都失败，返回默认图像
        print(f"警告: 无法读取图像 {file_path}，使用默认图像")
        return np.zeros((512, 512, 3), dtype=np.uint8)

    except Exception as e:
        print(f"错误: 读取图像 {file_path} 时发生异常: {e}")
        return np.zeros((512, 512, 3), dtype=np.uint8)


def get_data(path):
    """统一的数据加载函数"""
    data = {}
    for root, dirs, files in os.walk(path, topdown=False):
        for name in dirs:
            full_path = os.path.join(root, name)
            data[name] = {"path": full_path}

            # 收集所有图片文件
            all_files = []

            # 遍历所有子目录和文件
            for sub_root, sub_dirs, sub_files in os.walk(full_path):
                for file in sub_files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        # 获取相对于主目录的相对路径
                        rel_path = os.path.relpath(os.path.join(sub_root, file), full_path)
                        all_files.append(rel_path)

            data[name]["files"] = all_files

    return data


class SUES200DatasetTrain(Dataset):

    def __init__(self,
                 query_folder,
                 gallery_folder,
                 transforms_query=None,
                 transforms_gallery=None,
                 prob_flip=0.5,
                 shuffle_batch_size=128):
        super().__init__()

        print("Loading query data...")
        self.query_dict = get_data(query_folder)
        print("Loading gallery data...")
        self.gallery_dict = get_data(gallery_folder)

        # use only folders that exists for both gallery and query
        self.ids = list(set(self.query_dict.keys()).intersection(self.gallery_dict.keys()))
        self.ids.sort()

        print(f"Found {len(self.ids)} common IDs")

        self.pairs = []

        for idx in self.ids:
            # 检查query文件
            if len(self.query_dict[idx]["files"]) == 0:
                print(f"Warning: No query files found for ID {idx}")
                continue

            # query图像（satellite）- 使用第一个文件
            query_file = self.query_dict[idx]["files"][0]
            query_img_path = os.path.join(self.query_dict[idx]["path"], query_file)

            gallery_path = self.gallery_dict[idx]["path"]
            gallery_imgs = self.gallery_dict[idx]["files"]

            if len(gallery_imgs) == 0:
                print(f"Warning: No gallery files found for ID {idx}")
                continue

            for g_file in gallery_imgs:
                gallery_img_path = os.path.join(gallery_path, g_file)
                # 验证文件是否存在
                if not os.path.exists(query_img_path):
                    print(f"Warning: Query file does not exist: {query_img_path}")
                    continue
                if not os.path.exists(gallery_img_path):
                    print(f"Warning: Gallery file does not exist: {gallery_img_path}")
                    continue

                self.pairs.append((idx, query_img_path, gallery_img_path))

        print(f"Total valid pairs created: {len(self.pairs)}")

        self.transforms_query = transforms_query
        self.transforms_gallery = transforms_gallery
        self.prob_flip = prob_flip
        self.shuffle_batch_size = shuffle_batch_size

        self.samples = copy.deepcopy(self.pairs)

    def __getitem__(self, index):

        idx, query_img_path, gallery_img_path = self.samples[index]

        # 使用安全的图像读取方法
        query_img = safe_imread(query_img_path)
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)

        gallery_img = safe_imread(gallery_img_path)
        gallery_img = cv2.cvtColor(gallery_img, cv2.COLOR_BGR2RGB)

        if np.random.random() < self.prob_flip:
            query_img = cv2.flip(query_img, 1)
            gallery_img = cv2.flip(gallery_img, 1)

            # image transforms
        if self.transforms_query is not None:
            query_img = self.transforms_query(image=query_img)['image']

        if self.transforms_gallery is not None:
            gallery_img = self.transforms_gallery(image=gallery_img)['image']

        return query_img, gallery_img, idx

    def __len__(self):
        return len(self.samples)

    def shuffle(self, ):
        '''
        custom shuffle function for unique class_id sampling in batch
        '''

        print("\nShuffle Dataset:")

        pair_pool = copy.deepcopy(self.pairs)

        # Shuffle pairs order
        random.shuffle(pair_pool)

        # Lookup if already used in epoch
        pairs_epoch = set()
        idx_batch = set()

        # buckets
        batches = []
        current_batch = []

        # counter
        break_counter = 0

        # progressbar
        pbar = tqdm()

        while True:

            pbar.update()

            if len(pair_pool) > 0:
                pair = pair_pool.pop(0)

                idx, _, _ = pair

                if idx not in idx_batch and pair not in pairs_epoch:

                    idx_batch.add(idx)
                    current_batch.append(pair)
                    pairs_epoch.add(pair)

                    break_counter = 0

                else:
                    # if pair fits not in batch and is not already used in epoch -> back to pool
                    if pair not in pairs_epoch:
                        pair_pool.append(pair)

                    break_counter += 1

                if break_counter >= 512:
                    break

            else:
                break

            if len(current_batch) >= self.shuffle_batch_size:
                # empty current_batch bucket to batches
                batches.extend(current_batch)
                idx_batch = set()
                current_batch = []

        pbar.close()

        # wait before closing progress bar
        time.sleep(0.3)

        self.samples = batches

        print("Original Length: {} - Length after Shuffle: {}".format(len(self.pairs), len(self.samples)))
        print("Break Counter:", break_counter)
        print("Pairs left out of last batch to avoid creating noise:", len(self.pairs) - len(self.samples))
        if len(self.samples) > 0:
            print("First Element ID: {} - Last Element ID: {}".format(self.samples[0][0], self.samples[-1][0]))


class SUES200DatasetEval(Dataset):

    def __init__(self,
                 data_folder,
                 mode,
                 transforms=None,
                 sample_ids=None,
                 gallery_n=-1):
        super().__init__()

        print(f"Loading {mode} data...")
        self.data_dict = get_data(data_folder)

        # use only folders that exists for both gallery and query
        self.ids = list(self.data_dict.keys())
        print(f"Found {len(self.ids)} IDs in {mode} data")

        self.transforms = transforms

        self.given_sample_ids = sample_ids

        self.images = []
        self.sample_ids = []

        self.mode = mode

        self.gallery_n = gallery_n

        for i, sample_id in enumerate(self.ids):
            files = self.data_dict[sample_id]["files"]
            if len(files) == 0:
                print(f"Warning: No files found for ID {sample_id} in {mode}")
                continue

            for j, file in enumerate(files):
                img_path = os.path.join(self.data_dict[sample_id]["path"], file)
                # 验证文件是否存在
                if not os.path.exists(img_path):
                    print(f"Warning: File does not exist: {img_path}")
                    continue

                self.images.append(img_path)
                self.sample_ids.append(sample_id)

        print(f"Total {mode} images: {len(self.images)}")

    def __getitem__(self, index):

        img_path = self.images[index]
        sample_id = self.sample_ids[index]

        # 使用安全的图像读取方法
        img = safe_imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # image transforms
        if self.transforms is not None:
            img = self.transforms(image=img)['image']

        label = int(sample_id)
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
    val_transforms = A.Compose([A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_AREA, p=1.0),
                                A.Normalize(mean, std),
                                ToTensorV2(),
                                ])

    # 修复Albumentations参数问题
    train_sat_transforms = A.Compose([
        A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_AREA, p=1.0),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, p=0.5),
        A.OneOf([
            A.Blur(blur_limit=3, p=1.0),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
        ], p=0.3),
        A.OneOf([
            A.GridDropout(ratio=0.4, p=1.0),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=1.0),
        ], p=0.3),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean, std),
        ToTensorV2(),
    ])

    train_drone_transforms = A.Compose([
        A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_AREA, p=1.0),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, p=0.5),
        A.OneOf([
            A.Blur(blur_limit=3, p=1.0),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
        ], p=0.3),
        A.OneOf([
            A.GridDropout(ratio=0.4, p=1.0),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=1.0),
        ], p=0.3),
        A.Normalize(mean, std),
        ToTensorV2(),
    ])

    return val_transforms, train_sat_transforms, train_drone_transforms


# 测试修复后的版本
if __name__ == "__main__":
    dataset = SUES200DatasetTrain(
        query_folder="D:/数据集/SUES-200-512x512-V2/SUES-200-512x512/train/satellite",
        gallery_folder="D:/数据集/SUES-200-512x512-V2/SUES-200-512x512/train/drone"
    )

    print(f"Dataset length: {len(dataset)}")
    if len(dataset) > 0:
        # 测试前几个样本
        for i in range(min(3, len(dataset))):
            query, gallery, idx = dataset[i]
            print(f"Sample {i} - ID: {idx}, Query shape: {query.shape}, Gallery shape: {gallery.shape}")