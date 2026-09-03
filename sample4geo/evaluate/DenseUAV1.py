import torch
import numpy as np
from tqdm import tqdm
import gc
import os
import json
import math


def predict_with_paths(model, data_loader, input_id=None):
    """提取特征同时保留图像路径"""
    # 如果是DataParallel，提取原始模型
    if hasattr(model, 'module'):
        single_model = model.module
    else:
        single_model = model

    # 将模型设置为评估模式并移动到GPU
    single_model.eval()
    single_model = single_model.cuda()

    features = []
    labels = []
    paths = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="提取特征")):
            # 假设batch返回 (inputs, labels, paths) 或 (inputs, labels)
            if len(batch) == 3:
                inputs, batch_labels, batch_paths = batch
            elif len(batch) == 4:  # 如果有额外的ID信息
                inputs, batch_labels, batch_paths, _ = batch
            else:
                inputs, batch_labels = batch

                # 尝试从数据集获取路径
                if hasattr(data_loader.dataset, 'images'):
                    # 计算当前批次的起始索引
                    start_idx = batch_idx * data_loader.batch_size
                    end_idx = start_idx + len(batch_labels)
                    batch_paths = data_loader.dataset.images[start_idx:end_idx]
                else:
                    batch_paths = []

            inputs = inputs.cuda()

            # 处理输入ID参数
            if input_id is not None:
                # 确保input_id也在GPU上
                input_id_gpu = input_id.cuda() if isinstance(input_id, torch.Tensor) else input_id
                batch_features = single_model(inputs, input_id=input_id_gpu)
            else:
                batch_features = single_model(inputs)

            # 确保特征在CPU上并转换为连续内存布局
            batch_features = batch_features.cpu().contiguous()

            features.append(batch_features)
            labels.append(batch_labels.cpu())

            # 如果从数据集获取到了路径，使用这些路径
            if batch_paths:
                paths.extend(batch_paths)
            # 否则尝试从数据集的images属性获取
            elif hasattr(data_loader.dataset, 'images'):
                # 如果是最后一个批次，可能长度不完整，所以动态计算
                if batch_idx == 0:
                    dataset_paths = data_loader.dataset.images
                else:
                    # 对于后续批次，我们需要知道已经处理了多少图像
                    processed_count = len(paths)
                    remaining = len(data_loader.dataset.images) - processed_count
                    if remaining > 0:
                        next_paths = data_loader.dataset.images[processed_count:processed_count + len(batch_labels)]
                        paths.extend(next_paths)

    if features:
        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0)
    else:
        features = torch.empty(0)
        labels = torch.empty(0)

    # 如果没有获取到路径，尝试从数据集的images属性获取全部
    if not paths and hasattr(data_loader.dataset, 'images'):
        paths = data_loader.dataset.images

    # 如果还是没有路径，生成默认路径
    if not paths or len(paths) != len(labels):
        print(f"警告: 路径数量 ({len(paths) if paths else 0}) 与标签数量 ({len(labels)}) 不匹配")
        # 生成默认路径
        paths = [f"image_{i}" for i in range(len(labels))]

    return features, labels, paths

def compute_mAP(index, good_index, junk_index):
    """计算平均精度"""
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()

    if good_index.size == 0:  # 如果没有匹配项
        cmc[0] = -1
        return ap, cmc

    # 移除无效索引
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # 找到匹配项的位置
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    if len(rows_good) == 0:
        cmc[0] = -1
        return ap, cmc

    # 计算CMC
    cmc[rows_good[0]:] = 1

    # 计算平均精度
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc



def haversine_distance(lat1, lon1, lat2, lon2):
    """计算米制距离（用于MA指标）"""
    # 使用Haversine公式计算大圆距离
    EARTH_RADIUS = 6371000  # 地球半径，单位：米
    PI = math.pi

    # 将经纬度转换为弧度
    lat1_rad = lat1 * PI / 180.0
    lat2_rad = lat2 * PI / 180.0
    lon1_rad = lon1 * PI / 180.0
    lon2_rad = lon2 * PI / 180.0

    # 计算经纬度差
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Haversine公式
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = EARTH_RADIUS * c
    return distance




def euclidean_distance(query, gallery):
    """计算经纬度的欧氏距离 - 与原始代码一致"""
    query = np.array(query, dtype=np.float32)
    gallery = np.array(gallery, dtype=np.float32)

    # 原始代码中的计算方法
    A = gallery - query
    A_T = A.transpose()
    distance_matrix = np.matmul(A, A_T)  # 距离平方矩阵
    mask = np.eye(distance_matrix.shape[0], dtype=bool)
    distances_squared = distance_matrix[mask]
    distances = np.sqrt(distances_squared.reshape(-1))

    return distances


def evaluate_sdm_score(distances, K):
    """计算SDM分数 - 与原始代码一致"""
    if distances is None or len(distances) == 0:
        return 0.0

    if len(distances) < K:
        K = len(distances)

    # 检查距离是否为0
    if np.all(np.abs(distances[:K]) < 1e-10):
        return 1.0

    # 使用与原始代码完全一致的公式
    weight = np.ones(K) - np.array(range(0, K, 1)) / K

    # 原始代码中的公式: m2 = 1 / np.exp(distance*5e3)
    m2 = 1 / np.exp(distances[:K] * 5e3)

    m3 = m2 * weight
    result = np.sum(m3) / np.sum(weight)

    return result


def evaluate_query(qf, ql, qpath, gf, gl, gpaths, config_dict, sdm_K_values, recall_K_values, query_idx):
    """评估单个查询"""
    # 计算相似度分数
    score = gf @ qf.unsqueeze(-1)
    score = score.squeeze().cpu().numpy()

    # 按分数排序（从高到低）
    index = np.argsort(score)[::-1]

    # 找到匹配项和无效项
    good_index = np.argwhere(gl == ql)
    junk_index = np.argwhere(gl == -1)

    # 计算Recall指标
    ap, cmc = compute_mAP(index, good_index, junk_index)

    # 计算SDM指标
    sdm_scores = {}

    # 获取查询图像的地理位置
    query_loc = get_location_from_path(qpath, config_dict)

    # 调试前几个查询
    if query_idx < 3:
        print(f"\n查询 {query_idx}: {os.path.basename(qpath)}")
        print(f"查询位置: {query_loc}")
        print(f"查询标签: {ql}")
        print(f"Top-5 图库标签: {gl[index[:5]]}")

    if query_loc is not None:
        for K in sdm_K_values:
            if K <= len(index):
                # 获取topK图库图像的地理位置
                topk_indices = index[:K]
                gallery_locs = []

                for idx in topk_indices:
                    if idx < len(gpaths):
                        gallery_loc = get_location_from_path(gpaths[idx], config_dict)
                        gallery_locs.append(gallery_loc)
                    else:
                        gallery_locs.append(None)

                # 检查是否所有位置都有效
                if all(loc is not None for loc in gallery_locs):
                    # 使用与原始代码一致的距离计算方法
                    distances = euclidean_distance(query_loc, gallery_locs)

                    if query_idx < 3 and K == 1:
                        print(f"Top-1 距离: {distances[0]:.8f} (度)")
                        print(f"SDM分数: {evaluate_sdm_score(distances, K):.6f}")

                    sdm_scores[K] = evaluate_sdm_score(distances, K)
                else:
                    # 如果有位置信息缺失，设置SDM分数为0
                    sdm_scores[K] = 0.0
            else:
                sdm_scores[K] = 0.0
    else:
        # 如果查询位置信息缺失，设置所有SDM分数为0
        for K in sdm_K_values:
            sdm_scores[K] = 0.0

    return ap, cmc, sdm_scores


def get_location_from_path(img_path, config_dict):
    """从图像路径获取地理位置信息 - 简化为仅使用文件夹名"""
    try:
        # 提取文件夹名（地点ID）
        folder_name = os.path.basename(os.path.dirname(img_path))

        if folder_name in config_dict:
            return config_dict[folder_name]
        else:
            # 尝试查找相似的键
            for key in config_dict.keys():
                if key == folder_name or key.endswith(folder_name) or folder_name.endswith(key):
                    return config_dict[key]
    except:
        pass

    return None


def load_gps_config(config_path):
    """加载GPS配置文件 - 仅使用文件夹名作为键"""
    config_dict = {}

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 3:
                    # 解析格式: path longitude latitude height
                    path = parts[0]  # 例如: train/satellite/000000/H80.tif

                    # 提取文件夹名作为键
                    folder_name = os.path.basename(os.path.dirname(path))

                    # 提取经度和纬度
                    try:
                        longitude = float(parts[1].replace("E", ""))
                        latitude = float(parts[2].replace("N", ""))

                        # 使用文件夹名作为键
                        config_dict[folder_name] = [longitude, latitude]

                    except ValueError:
                        continue

    print(f"GPS配置加载完成，共 {len(config_dict)} 条记录")

    # 显示一些样本用于验证
    sample_keys = list(config_dict.keys())[:3]
    for key in sample_keys:
        print(f"  地点 {key}: 经度={config_dict[key][0]:.6f}, 纬度={config_dict[key][1]:.6f}")

    return config_dict


def evaluate(config, model, query_loader, gallery_loader,
             ranks=[1, 5, 10], sdm_K=[1, 3, 5, 10],
             step_size=1000, cleanup=True, input_id=None):
    """
    评估DenseUAV数据集
    """

    # 加载GPS配置
    gps_config_path = os.path.join(config.data_folder, "Dense_GPS_ALL.txt")
    config_dict = load_gps_config(gps_config_path)

    print(f"加载了 {len(config_dict)} 条GPS信息")

    # 提取特征和路径
    print("提取查询集特征...")
    qf, ql, qpaths = predict_with_paths(model, query_loader, input_id=input_id)

    print("提取图库集特征...")
    gf, gl, gpaths = predict_with_paths(model, gallery_loader, input_id=input_id)

    # 检查位置信息匹配情况
    print("\n检查位置信息匹配情况:")
    found_locations = 0
    for i, path in enumerate(qpaths[:10]):
        loc = get_location_from_path(path, config_dict)
        if loc is not None:
            found_locations += 1
            print(f"  查询 {i}: {os.path.basename(path)} -> 找到位置")
        else:
            print(f"  查询 {i}: {os.path.basename(path)} -> 未找到位置")

    print(f"\n前10个查询中，找到位置信息的: {found_locations}/10")

    if found_locations == 0:
        print("警告：无法找到任何位置信息！")
        print("可能原因:")
        print("  1. GPS配置文件路径不正确")
        print("  2. 图像路径格式与GPS记录不匹配")
        print("  3. GPS文件格式有误")
        return 0.0

    # 将特征转移到GPU
    qf = qf.cuda()
    gf = gf.cuda()

    # 确保特征是连续的
    qf = qf.contiguous()
    gf = gf.contiguous()

    # 初始化结果存储
    total_queries = len(ql)
    CMC = torch.IntTensor(len(gl)).zero_()
    ap_total = 0.0
    valid_queries = 0

    sdm_results = {K: 0.0 for K in sdm_K}
    sdm_counts = {K: 0 for K in sdm_K}

    # 转换为numpy数组用于计算
    gl_np = gl.numpy()

    print("评估每个查询...")
    for i in tqdm(range(total_queries), desc="评估查询"):
        # 仅评估前50个查询以加快速度（调试用）
        # if i >= 50:
        #     break

        ap, cmc, sdm_scores = evaluate_query(
            qf[i], ql[i], qpaths[i],
            gf, gl_np, gpaths,
            config_dict, sdm_K, ranks, i
        )

        if cmc[0] == -1:
            continue

        CMC = CMC + cmc
        ap_total += ap
        valid_queries += 1

        # 累加SDM分数
        for K, score in sdm_scores.items():
            sdm_results[K] += score
            sdm_counts[K] += 1

    # 计算平均指标
    if valid_queries > 0:
        AP = ap_total / valid_queries * 100
        CMC = CMC.float() / valid_queries * 100
    else:
        AP = 0
        CMC = torch.zeros_like(CMC.float())

    # 计算平均SDM
    avg_sdm = {}
    for K in sdm_K:
        if sdm_counts[K] > 0:
            avg_sdm[K] = sdm_results[K] / sdm_counts[K] * 100
        else:
            avg_sdm[K] = 0.0

    # 输出结果
    print("\n" + "=" * 60)
    print("DenseUAV 评估结果:")
    print("=" * 60)

    print(f"查询总数: {total_queries}")
    print(f"有效查询: {valid_queries}")
    print(f"有位置信息的查询: {sdm_counts[1]}/{total_queries} ({sdm_counts[1] / total_queries * 100:.1f}%)")

    print("\nRecall指标:")
    for K in ranks:
        if K - 1 < len(CMC):
            recall = CMC[K - 1].item()
            print(f"  Recall@{K}: {recall:.2f}%")

    print(f"\nmAP: {AP:.2f}%")

    print("\nSDM指标 (仅计算有位置信息的查询):")
    for K, score in avg_sdm.items():
        print(f"  SDM@{K}: {score:.2f}%")

    print("\n理论分析:")
    print(f"  - Recall@1: {CMC[0].item():.2f}% (精确匹配的比例)")
    print(f"  - SDM@1: {avg_sdm[1]:.2f}% (平均地理相似度)")
    print(f"  - SDM@1应≥Recall@1，因为:")
    print(f"    1. 匹配正确时，SDM=1")
    print(f"    2. 匹配错误时，SDM>0")
    print(f"    3. 因此平均值应≥Recall@1")

    if avg_sdm[1] < CMC[0].item():
        print("\n警告: SDM@1 < Recall@1，这可能表明:")
        print("  1. 位置信息匹配不正确")
        print("  2. 距离计算有误")
        print("  3. 很多查询没有位置信息")

    print("\n" + "=" * 60)

    # 清理内存
    if cleanup:
        del qf, gf, ql, gl
        gc.collect()
        torch.cuda.empty_cache()

    # 返回Recall@1作为主要指标
    return CMC[0].item() if 0 < len(CMC) else 0.0