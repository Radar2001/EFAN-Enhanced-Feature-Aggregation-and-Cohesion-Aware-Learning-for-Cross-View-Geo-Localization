import torch
import numpy as np
from mpmath.math2 import sqrt2
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


def load_gps_config(config_path):
    """加载GPS配置文件"""
    config_dict = {}

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 3:
                    # 解析格式: path longitude latitude height
                    path = parts[0]  # 例如: train/satellite/000000/H80.tif

                    # 提取经度和纬度
                    try:
                        # 注意：经度是E开头，纬度是N开头
                        longitude_str = parts[1]
                        latitude_str = parts[2]

                        # 移除E和N前缀，转换为浮点数
                        longitude = float(longitude_str.replace("E", ""))
                        latitude = float(latitude_str.replace("N", ""))

                        # 从路径中提取关键信息
                        # 使用文件名（不含扩展名）或文件夹名作为键
                        file_name = os.path.basename(path)
                        file_name_no_ext = os.path.splitext(file_name)[0]
                        folder_name = os.path.basename(os.path.dirname(path))

                        # 使用完整相对路径作为主要键
                        config_dict[path] = [longitude, latitude]

                        # 同时使用文件名作为备用键
                        config_dict[file_name_no_ext] = [longitude, latitude]

                        # 同时使用文件夹名作为备用键
                        config_dict[folder_name] = [longitude, latitude]

                    except ValueError as e:
                        print(f"解析GPS行失败: {line.strip()}, 错误: {e}")
                        continue

    print(f"GPS配置加载完成，共 {len(config_dict)} 条记录")
    # 打印前5条记录用于调试
    for i, (key, value) in enumerate(list(config_dict.items())[:5]):
        print(f"  {key}: {value}")

    return config_dict


def get_location_from_path(img_path, config_dict):
    """从图像路径获取地理位置信息"""
    # 方法1: 尝试完整路径匹配（去除根目录）
    # 假设img_path是绝对路径或相对路径，我们需要提取相对路径部分
    rel_path = img_path
    for base_path in ["train/", "test/", "query_", "gallery_", "drone/", "satellite/"]:
        if base_path in img_path:
            # 找到base_path之后的部分
            idx = img_path.find(base_path)
            if idx != -1:
                rel_path = img_path[idx:]
                break

    if rel_path in config_dict:
        return config_dict[rel_path]

    # 方法2: 尝试使用文件名（不含扩展名）
    file_name = os.path.basename(img_path)
    file_name_no_ext = os.path.splitext(file_name)[0]

    if file_name_no_ext in config_dict:
        return config_dict[file_name_no_ext]

    # 方法3: 尝试使用文件夹名
    folder_name = os.path.basename(os.path.dirname(img_path))
    if folder_name in config_dict:
        return config_dict[folder_name]

    # 方法4: 尝试在路径中查找任何匹配的键
    for key in config_dict.keys():
        if key in img_path:
            return config_dict[key]

    # 调试：打印找不到位置的路径
    print(f"警告: 找不到图像 {img_path} 的位置信息")
    print(f"  尝试的路径: {rel_path}")
    print(f"  文件名: {file_name_no_ext}")
    print(f"  文件夹名: {folder_name}")

    return None


def calculate_geographic_distance(loc1, loc2):
    """计算两个地理位置之间的距离（米）"""
    if loc1 is None or loc2 is None:
        return None

    # 使用Haversine公式计算大圆距离
    return latlog2meter(loc1[1], loc1[0], loc2[1], loc2[0])


def latlog2meter(lat1, lon1, lat2, lon2):
    """将经纬度转换为米制距离""" #此步不需要，需要修改
    # EARTH_RADIUS = 6378.137 * 1000  # 转换为米
    # PI = math.pi
    #
    # # 转换为弧度
    # lat1_rad = lat1 * PI / 180
    # lat2_rad = lat2 * PI / 180
    # lon1_rad = lon1 * PI / 180
    # lon2_rad = lon2 * PI / 180

    # 计算经纬度差
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine公式
    a = math.sqrt((dlat) ** 2 + (dlon) ** 2)
    # c = 2 * math.asin(min(1, math.sqrt(a)))

    # distance = EARTH_RADIUS * c
    return a


def evaluate_sdm_score(distances, K):
    """计算SDM分数"""
    if distances is None:
        return 0.0

    if len(distances) < K:
        K = len(distances)

    # 检查距离是否为0
    if np.all(np.abs(distances[:K]) < 1e-10):
        # print(f"警告：前{K}个距离都接近0: {distances[:K]}")
        return 1.0

    # 使用原始论文中的SDM计算公式
    weight = np.ones(K) - np.array(range(0, K, 1))/K

    # 使用距离的倒数并进行指数衰减
    # 注意：距离单位是米，需要适当缩放
    scaled_distances = distances[:K]   # 转换为公里

    # weight = np.ones(K) - np.array(range(0, K, 1))/K


    # 使用指数衰减函数
    m2 = 1 / np.exp(scaled_distances * 5000)  # 使用5作为衰减系数

    m3 = m2 * weight
    result = np.sum(m3) / np.sum(weight)

    # 调试信息
    # if result > 0.99:
    #     print(f"SDM分数过高: {result:.4f}")
    #     print(f"  距离: {distances[:K]}")
    #     print(f"  缩放距离: {scaled_distances}")
    #     print(f"  m2: {m2}")

    return result


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


def evaluate_query(qf, ql, qpath, gf, gl, gpaths, config_dict, sdm_K_values, recall_K_values, debug=False):
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

    if debug:
        print(f"\n调试查询: {qpath}")
        print(f"查询位置: {query_loc}")

    if query_loc is not None:
        for K in sdm_K_values:
            if K <= len(index):
                # 获取topK图库图像的地理位置
                topk_indices = index[:K]
                distances = []

                for idx in topk_indices:
                    gallery_loc = get_location_from_path(gpaths[idx], config_dict)
                    if query_loc is not None and gallery_loc is not None:
                        dist = latlog2meter(query_loc[1], query_loc[0],
                                            gallery_loc[1], gallery_loc[0])
                        distances.append(dist)
                    else:
                        distances.append(float('inf'))

                distances = np.array(distances, dtype=np.float32)

                # if debug:
                #     print(f"SDM@{K} 距离: {distances}")

                sdm_scores[K] = evaluate_sdm_score(distances, K)
            else:
                sdm_scores[K] = 0.0
    else:
        # 如果查询位置信息缺失，设置所有SDM分数为0
        for K in sdm_K_values:
            sdm_scores[K] = 0.0

    # 计算MA指标（米制精度）
    ma_scores = {}
    if query_loc is not None and len(index) > 0:
        top1_idx = index[0]
        gallery_loc = get_location_from_path(gpaths[top1_idx], config_dict)

        if gallery_loc is not None:
            distance_meter = latlog2meter(query_loc[1], query_loc[0],
                                          gallery_loc[1], gallery_loc[0])

            # if debug:
            #     print(f"MA距离: {distance_meter}米")

            # 多个阈值
            for meter_threshold in [1, 5, 10, 20, 50, 100]:
                ma_scores[meter_threshold] = 1.0 if distance_meter < meter_threshold else 0.0
        else:
            # 如果图库位置信息缺失，设置所有MA分数为0
            for meter_threshold in [1, 5, 10, 20, 50, 100]:
                ma_scores[meter_threshold] = 0.0
    else:
        # 如果查询或图库位置信息缺失，设置所有MA分数为0
        for meter_threshold in [1, 5, 10, 20, 50, 100]:
            ma_scores[meter_threshold] = 0.0

    return ap, cmc, sdm_scores, ma_scores


def evaluate(config, model, query_loader, gallery_loader,
             ranks=[1, 5, 10], sdm_K=[1, 3, 5, 10],
             step_size=1000, cleanup=True, input_id=None):
    """
    评估DenseUAV数据集

    参数:
        config: 配置对象，需要包含:
            - data_folder: 数据集根目录
            - mode: 评估模式 ('1': drone->satellite, '2': satellite->drone)
        model: 特征提取模型
        query_loader: 查询集数据加载器
        gallery_loader: 图库集数据加载器
        ranks: Recall评估的K值列表
        sdm_K: SDM评估的K值列表
        step_size: 特征提取的批处理大小
        cleanup: 是否清理内存
        input_id: 输入ID参数（用于多模态模型）
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

    # 检查是否有位置信息
    query_locs = [get_location_from_path(path, config_dict) for path in qpaths[:10]]  # 只检查前10个
    gallery_locs = [get_location_from_path(path, config_dict) for path in gpaths[:10]]

    query_with_loc = sum(1 for loc in query_locs if loc is not None)
    gallery_with_loc = sum(1 for loc in gallery_locs if loc is not None)

    print(f"前10个查询图像中有位置信息的: {query_with_loc}/10")
    print(f"前10个图库图像中有位置信息的: {gallery_with_loc}/10")

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
    ma_results = {meter: 0.0 for meter in [1, 5, 10, 20, 50, 100]}

    sdm_counts = {K: 0 for K in sdm_K}
    ma_counts = {meter: 0 for meter in [1, 5, 10, 20, 50, 100]}

    # 转换为numpy数组用于计算
    gl_np = gl.numpy()

    print("评估每个查询...")
    for i in tqdm(range(total_queries), desc="评估查询"):
        # 前3个查询开启调试模式
        debug_mode = (i < 3)

        ap, cmc, sdm_scores, ma_scores = evaluate_query(
            qf[i], ql[i], qpaths[i],
            gf, gl_np, gpaths,
            config_dict, sdm_K, ranks, debug=debug_mode
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

        # 累加MA分数
        for meter, score in ma_scores.items():
            ma_results[meter] += score
            ma_counts[meter] += 1

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

    # 计算平均MA
    avg_ma = {}
    for meter in ma_results.keys():
        if ma_counts[meter] > 0:
            avg_ma[meter] = ma_results[meter] / ma_counts[meter] * 100
        else:
            avg_ma[meter] = 0.0

    # 输出结果
    print("\n" + "=" * 60)
    print("DenseUAV 评估结果:")
    print("=" * 60)

    print(f"查询总数: {total_queries}")
    print(f"有效查询: {valid_queries}")

    print("\nRecall指标:")
    recall_strings = []
    for K in ranks:
        if K - 1 < len(CMC):
            recall = CMC[K - 1].item()
            recall_strings.append(f"Recall@{K}: {recall:.2f}%")
            print(f"  Recall@{K}: {recall:.2f}%")

    print(f"\nmAP: {AP:.2f}%")

    print("\nSDM指标:")
    for K, score in avg_sdm.items():
        print(f"  SDM@{K}: {score:.2f}%")

    print("\nMA指标 (米制精度):")
    for meter, score in avg_ma.items():
        print(f"  MA@{meter}m: {score:.2f}%")

    print("\n" + "=" * 60)

    # 保存结果到JSON
    # results = {
    #     "Recall": {f"@{K}": CMC[K - 1].item() if K - 1 < len(CMC) else 0.0 for K in ranks},
    #     "mAP": AP,
    #     "SDM": avg_sdm,
    #     "MA": avg_ma,
    #     "mode": getattr(config, 'mode', '1'),
    #     "total_queries": total_queries,
    #     "valid_queries": valid_queries,
    #     "query_with_gps": query_with_loc,
    #     "gallery_with_gps": gallery_with_loc
    # }
    #
    # # 根据模式确定输出文件名
    # mode_str = "drone2satellite" if getattr(config, 'mode', '1') == "1" else "satellite2drone"
    # output_file = f"DenseUAV_eval_{mode_str}.json"
    #
    # with open(output_file, 'w') as f:
    #     json.dump(results, f, indent=4)
    #
    # print(f"\n详细结果已保存到: {output_file}")

    # 清理内存
    if cleanup:
        del qf, gf, ql, gl
        gc.collect()
        torch.cuda.empty_cache()

    # 返回Recall@1作为主要指标（与university.py保持一致）
    return CMC[0].item() if 0 < len(CMC) else 0.0