import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.nn


class InfoNCE(nn.Module):

    def __init__(self, loss_function, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()

        self.loss_function = loss_function
        self.device = device

    def forward(self, image_features1, image_features2, logit_scale):
        image_features1 = F.normalize(image_features1, dim=-1)
        image_features2 = F.normalize(image_features2, dim=-1)

        logits_per_image1 = logit_scale * image_features1 @ image_features2.T

        logits_per_image2 = logits_per_image1.T

        labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=self.device)

        loss = (self.loss_function(logits_per_image1, labels) + self.loss_function(logits_per_image2, labels)) / 2

        return loss
class InfoNCE_interself(nn.Module):

    def __init__(self, loss_function, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()

        self.loss_function = loss_function
        self.device = device

    def forward(self, image_features1, image_features2, logit_scale):
        image_features1 = F.normalize(image_features1, dim=-1)
        image_features2 = F.normalize(image_features2, dim=-1)

        logits_per_image1 = logit_scale * image_features1 @ image_features2.T

        logits_per_image2 = logits_per_image1.T

        logits_per_image1_self = logit_scale * image_features1 @ image_features1.T
        logits_per_image2_self = logit_scale * image_features2 @ image_features2.T

        labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=self.device)

        loss = (self.loss_function(logits_per_image1, labels) + self.loss_function(logits_per_image2, labels)) / 2 +(self.loss_function(logits_per_image1_self, labels) + self.loss_function(logits_per_image2_self, labels))/2/4

        return loss

class InfoNCE_inter(nn.Module):

    def __init__(self, loss_function, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()

        self.loss_function = loss_function
        self.device = device
        self.delta_n = 0.9
        self.reduction = 'mean'

    def forward(self, image_features1, image_features2, logit_scale):
        image_features1 = F.normalize(image_features1, dim=-1)
        image_features2 = F.normalize(image_features2, dim=-1)

        logits_per_image1 = logit_scale * image_features1 @ image_features2.T

        logits_per_image2 = logits_per_image1.T

        logits_image1_1 = (image_features1 @ image_features1.T)
        logits_image2_1 = (image_features2 @ image_features2.T)

        loss1_1 = self._circle_loss_single(logits_image1_1, logit_scale)
        loss2_1 = self._circle_loss_single(logits_image2_1, logit_scale)

        # labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=self.device)
        # 计算对称损失
        labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=self.device)

        loss = (self.loss_function(logits_per_image1, labels) + self.loss_function(logits_per_image2,
                                                                                   labels)) / 2 + (
                       loss1_1 + loss2_1) / 2 * 0.25

        return loss

    def _circle_loss_single(self, similarity_matrix, logit_scale):
        """
        计算单方向的Circle Loss

        参数:
            similarity_matrix: 相似度矩阵 [batch_size, batch_size]
                              diagonal[i, i] 是正样本相似度
                              off-diagonal[i, j (j≠i)] 是负样本相似度

        返回:
            circle loss
        """

        batch_size = similarity_matrix.size(0)

        # 创建掩码：对角线为正样本，非对角线为负样本

        pos_mask = torch.eye(batch_size, dtype=torch.bool, device=self.device)
        neg_mask = ~pos_mask

        # 正样本相似度 (s_p)
        # pos_similarities = similarity_matrix[pos_mask].unsqueeze(1)  # [batch_size, 1]

        # 负样本相似度 (s_n)
        neg_similarities = similarity_matrix[neg_mask].view(batch_size, batch_size - 1)  # [batch_size, batch_size-1]

        # 计算自适应权重 alpha_p 和 alpha_n
        # alpha_p = relu(O_p - s_p), 其中 O_p = delta_p
        # alpha_n = relu(s_n - O_n), 其中 O_n = delta_n
        # alpha_p = torch.relu(self.delta_p - pos_similarities.detach())
        alpha_n = torch.relu(neg_similarities.detach() - self.delta_n)

        # 计算正样本项的logsumexp
        # log(1 + sum(exp(-gamma * alpha_p * (s_p - delta_p))))
        # pos_term = -logit_scale * alpha_p * (pos_similarities - self.delta_p)
        # log_pos = torch.logsumexp(pos_term, dim=1, keepdim=True)  # [batch_size, 1]

        # 计算负样本项的logsumexp
        # log(1 + sum(exp(gamma * alpha_n * (s_n - delta_n))))
        neg_term = logit_scale * alpha_n * (neg_similarities - self.delta_n)
        log_neg = torch.logsumexp(neg_term, dim=1, keepdim=True)  # [batch_size, 1]

        # Circle Loss公式: log(1 + exp(...)) 的稳定计算
        # 使用 logsumexp 技巧避免数值溢出
        # loss_per_sample = torch.log1p(torch.exp(log_neg + log_pos))
        soft_plus = nn.Softplus()
        loss_per_sample = soft_plus(log_neg)
        # 根据reduction参数聚合损失
        if self.reduction == 'mean':
            return loss_per_sample.mean()
        elif self.reduction == 'sum':
            return loss_per_sample.sum()
        else:
            return loss_per_sample


class CircleLoss(nn.Module):
    """
    Circle Loss: A Unified Perspective of Pair Similarity Optimization
    https://arxiv.org/abs/2002.10857

    对称版本的Circle Loss，适用于对比学习场景
    """

    def __init__(self,
                 m: float = 0.25,
                 gamma: float = 256.0,  # 缩放因子，相当于1/τ
                 delta_p=0.75,
                 delta_n=0.25,
                 reduction: str = 'mean',
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        参数:
            m: 间隔参数，控制正负样本相似度的目标间隔
            gamma: 缩放因子，控制梯度幅度（类似于1/温度τ）
            delta_p: 正样本最优相似度目标 (1-m)
            delta_n: 负样本最优相似度目标 (m)
            reduction: 'mean'或'sum'，指定损失减少方式
            device: 计算设备
        """
        super().__init__()

        self.m = m
        self.gamma = gamma
        self.delta_p = delta_p
        self.delta_n = delta_n
        self.reduction = reduction
        self.device = device

        # 验证参数
        assert delta_p > delta_n, f"delta_p ({delta_p}) must be greater than delta_n ({delta_n})"
        assert reduction in ['mean', 'sum'], "reduction must be 'mean' or 'sum'"

    def forward(self, image_features1, image_features2, logit_scale):
        """
        前向传播

        参数:
            image_features1: 第一组特征 [batch_size, feature_dim]
            image_features2: 第二组特征 [batch_size, feature_dim]
            logit_scale: 可学习的缩放参数，通常用于控制相似度范围

        返回:
            对称Circle Loss
        """
        # 归一化特征向量
        image_features1 = F.normalize(image_features1, dim=-1)
        image_features2 = F.normalize(image_features2, dim=-1)

        # 计算相似度矩阵
        # logits_per_image1[i, j] = similarity(image1[i], image2[j])
        logits_per_image1 = (image_features1 @ image_features2.T)

        # 对称版本：交换顺序
        logits_per_image2 = logits_per_image1.T

        # labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=self.device)
        # 计算对称损失
        loss1 = self._circle_loss_single(logits_per_image1, logit_scale)
        loss2 = self._circle_loss_single(logits_per_image2, logit_scale)

        # 对称损失平均
        loss = (loss1 + loss2) / 2

        return loss

    def _circle_loss_single(self, similarity_matrix, logit_scale):
        """
        计算单方向的Circle Loss

        参数:
            similarity_matrix: 相似度矩阵 [batch_size, batch_size]
                              diagonal[i, i] 是正样本相似度
                              off-diagonal[i, j (j≠i)] 是负样本相似度

        返回:
            circle loss
        """
        batch_size = similarity_matrix.size(0)

        # 创建掩码：对角线为正样本，非对角线为负样本

        pos_mask = torch.eye(batch_size, dtype=torch.bool, device=self.device)
        neg_mask = ~pos_mask

        # 正样本相似度 (s_p)
        pos_similarities = similarity_matrix[pos_mask].unsqueeze(1)  # [batch_size, 1]

        # 负样本相似度 (s_n)
        neg_similarities = similarity_matrix[neg_mask].view(batch_size, batch_size - 1)  # [batch_size, batch_size-1]

        # 计算自适应权重 alpha_p 和 alpha_n
        # alpha_p = relu(O_p - s_p), 其中 O_p = delta_p
        # alpha_n = relu(s_n - O_n), 其中 O_n = delta_n
        alpha_p = torch.relu(self.delta_p - pos_similarities.detach())
        alpha_n = torch.relu(neg_similarities.detach() - self.delta_n - alpha_p.detach().mean())

        # 计算正样本项的logsumexp
        # log(1 + sum(exp(-gamma * alpha_p * (s_p - delta_p))))
        pos_term = -self.gamma * alpha_p * (pos_similarities - self.delta_p)
        log_pos = torch.logsumexp(pos_term, dim=1, keepdim=True)  # [batch_size, 1]

        # 计算负样本项的logsumexp
        # log(1 + sum(exp(gamma * alpha_n * (s_n - delta_n))))
        neg_term = self.gamma * alpha_n * (neg_similarities - self.delta_n)
        log_neg = torch.logsumexp(neg_term, dim=1, keepdim=True)  # [batch_size, 1]

        # Circle Loss公式: log(1 + exp(...)) 的稳定计算
        # 使用 logsumexp 技巧避免数值溢出
        # loss_per_sample = torch.log1p(torch.exp(log_neg + log_pos))
        soft_plus = nn.Softplus()
        loss_per_sample = soft_plus(log_neg + log_pos)
        # 根据reduction参数聚合损失
        if self.reduction == 'mean':
            return loss_per_sample.mean()
        elif self.reduction == 'sum':
            return loss_per_sample.sum()
        else:
            return loss_per_sample
        # return loss_per_sample


class AdaptiveHybridLoss(nn.Module):
    """
    根据训练状态动态调整Circle Loss和InfoNCE的权重
    核心思想：
    - 训练早期：主要使用InfoNCE，学习基础特征
    - 训练中期：逐渐增加Circle Loss权重，进行困难样本挖掘
    - 训练后期：如果检测到噪声（伪困难样本），降低Circle Loss权重
    """

    def __init__(self,
                 loss_function,
                 base_temperature=0.07,
                 circle_margin=0.25,
                 circle_gamma=256,
                 initial_circle_weight=0.3,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):

        super().__init__()

        self.loss_function = loss_function
        self.base_temperature = base_temperature
        self.circle_margin = circle_margin
        self.circle_gamma = circle_gamma
        self.device = device

        # 权重调度
        self.initial_circle_weight = initial_circle_weight
        self.current_epoch = 0

    def set_epoch(self, epoch):
        """在训练循环中设置当前epoch"""
        self.current_epoch = epoch

    def forward(self, drone_features, sat_features, logit_scale):
        """
        前向传播
        """

        if self.current_epoch == -1:
            # 只计算InfoNCE损失
            info_nce_loss_d2s = self.info_nce_loss(drone_features, sat_features, logit_scale, self.loss_function)
            info_nce_loss_s2d = self.info_nce_loss(sat_features, drone_features, logit_scale, self.loss_function)
            total_loss = (info_nce_loss_d2s + info_nce_loss_s2d) / 2
            return total_loss

            # 第二阶段：epoch>=1，使用混合损失
        else:
            # 计算噪声水平
            labels = torch.arange(len(drone_features), dtype=torch.long, device=self.device)

            noise_level_drone, sim_matrix_drone = self.compute_noise_level(drone_features, labels)
            noise_level_sat, sim_matrix_sat = self.compute_noise_level(sat_features, labels)

            # 2. 计算自适应权重
            info_nce_weight, circle_weight = self.compute_adaptive_weights((noise_level_drone + noise_level_sat) / 2)

            # 3. 计算两个损失
            # InfoNCE损失：无人机作为query，卫星作为key
            info_nce_loss_d2s = self.info_nce_loss(drone_features, sat_features, logit_scale, self.loss_function)

            # InfoNCE损失：卫星作为query，无人机作为key
            info_nce_loss_s2d = self.info_nce_loss(sat_features, drone_features, logit_scale, self.loss_function)

            info_nce_loss = (info_nce_loss_d2s + info_nce_loss_s2d) / 2

            # new Circle Loss
            circle_loss_d2s = self.circle_loss_pro(drone_features, sat_features, labels, sim_matrix_drone)
            circle_loss_s2d = self.circle_loss_pro(sat_features, drone_features, labels, sim_matrix_sat)
            circle_loss = (circle_loss_d2s + circle_loss_s2d) / 2

            # 4. 加权组合
            total_loss = info_nce_weight * info_nce_loss + circle_weight * circle_loss

        return total_loss #, debug_info

    def compute_noise_level(self, features, labels):
        """
        """

        features = F.normalize(features, dim=-1)
        # 计算相似度矩阵
        sim_matrix = features @ features.T

        # 标签矩阵
        label_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)

        # 非匹配样本的相似度分布
        neg_mask = ~label_matrix
        neg_sim = sim_matrix[neg_mask]


        # 如果有很多负样本的相似度较高，说明存在大量伪困难样本
        high_sim_ratio = (neg_sim > 0.8 ).float().mean()  #convnext中为0.75

        return high_sim_ratio.item(), sim_matrix

    def compute_adaptive_weights(self, noise_level):
        """
        根据噪声水平自适应调整权重
        """
        # 基础权重（根据训练进度）

        base_circle_weight = self.initial_circle_weight

        # progress = min(self.current_epoch / self.warmup_epochs, 1.0)
        # if self.current_epoch < self.warmup_epochs:
        #     progress = 0.0
        # else:
        #     progress = 1.0
        # base_circle_weight =  progress * self.initial_circle_weight

        # 根据噪声调整
        # 噪声越多，越依赖InfoNCE
        noise_adjustment = 1.0 - noise_level  # 噪声多时降低Circle Loss权重
        adaptive_circle_weight = base_circle_weight * noise_adjustment

        # 确保权重在合理范围内
        adaptive_circle_weight = max(0, adaptive_circle_weight)

        circle_weight = adaptive_circle_weight
        info_nce_weight = 1.0 - circle_weight

        return info_nce_weight, circle_weight

    def info_nce_loss(self, query, key, logit_scale,loss_function):
        """标准的InfoNCE损失"""
        # 归一化
        # query = F.normalize(query, p=2, dim=1)
        # key = F.normalize(key, p=2, dim=1)
        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)
        # 计算相似度矩阵
        logits = logit_scale*query @ key.T

        # 标签：对角线为正样本
        labels = torch.arange(len(logits), dtype=torch.long, device=self.device)
        loss= loss_function(logits, labels)
        return loss

    def circle_loss_pro(self, query, key, labels,noise_sim_matrix):
        """标准的Circle Loss"""
        # 归一化
        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)

        # 相似度矩阵
        sim_matrix = query @ key.T

        batch_size=sim_matrix.shape[0]
        # 标签矩阵
        pos_mask = torch.eye(batch_size, dtype=torch.bool, device=self.device)
        neg_mask = ~pos_mask
        noise_sim_matrix_mask=~pos_mask

        # 正样本相似度 (s_p)
        pos_similarities = sim_matrix[pos_mask].unsqueeze(1)  # [batch_size, 1]

        # 负样本相似度 (s_n)
        neg_similarities = sim_matrix[neg_mask].view(batch_size, batch_size - 1)  # [batch_size, batch_size-1]

        noise_sim = noise_sim_matrix[noise_sim_matrix_mask].view(batch_size, batch_size - 1)

        # 计算自适应权重 alpha_p 和 alpha_n
        # alpha_p = relu(O_p - s_p), 其中 O_p = delta_p
        # alpha_n = relu(s_n - O_n), 其中 O_n = delta_n
        alpha_p = torch.relu(1-self.circle_margin - pos_similarities.detach())
        alpha_n = torch.relu(neg_similarities.detach() - self.circle_margin )#- alpha_p.detach().mean()
        # alpha_noise=torch.relu(1 -noise_sim.detach())
        # alpha_noise = torch.relu(noise_sim.detach())
        # 计算正样本项的logsumexp
        # log(1 + sum(exp(-gamma * alpha_p * (s_p - delta_p))))
        pos_term = -self.circle_gamma * alpha_p * (pos_similarities - 1+self.circle_margin)
        log_pos = torch.logsumexp(pos_term, dim=1, keepdim=True)  # [batch_size, 1]

        # 计算负样本项的logsumexp
        # log(1 + sum(exp(gamma * alpha_n * (s_n - delta_n))))
        neg_term = self.circle_gamma * alpha_n * (neg_similarities - self.circle_margin)   # * alpha_noise
        log_neg = torch.logsumexp(neg_term, dim=1, keepdim=True)  # [batch_size, 1]

        # Circle Loss公式: log(1 + exp(...)) 的稳定计算
        # 使用 logsumexp 技巧避免数值溢出
        # loss_per_sample = torch.log1p(torch.exp(log_neg + log_pos))
        soft_plus = nn.Softplus()
        loss_per_sample = soft_plus(log_neg + log_pos)
        # 根据reduction参数聚合损失

        return loss_per_sample.mean()

        # # 正负样本相似度
        # pos_sim = sim_matrix[pos_mask.bool()]
        # neg_sim = sim_matrix[neg_mask.bool()]
        # noise_sim = noise_sim_matrix[noise_sim_matrix_mask.bool()]
        #
        # # alpha_p = torch.relu(self.delta_p - pos_similarities.detach())
        # # alpha_n = torch.relu(neg_similarities.detach() - self.delta_n - alpha_p.detach().mean())
        # #
        # # # 计算正样本项的logsumexp
        # # # log(1 + sum(exp(-gamma * alpha_p * (s_p - delta_p))))
        # # pos_term = -self.gamma * alpha_p * (pos_similarities - self.delta_p)
        # # log_pos = torch.logsumexp(pos_term, dim=1, keepdim=True)  # [batch_size, 1]
        # #
        # # # 计算负样本项的logsumexp
        # # # log(1 + sum(exp(gamma * alpha_n * (s_n - delta_n))))
        # # neg_term = self.gamma * alpha_n * (neg_similarities - self.delta_n)
        # # log_neg = torch.logsumexp(neg_term, dim=1, keepdim=True)  # [batch_size, 1]
        #
        # #等效于-m，即取m为负值了。改回来了，改为上限为1， 下线为0.25
        # # Circle Loss计算
        # alpha_p = F.relu(1-self.circle_margin - pos_sim.detach())
        # alpha_n = F.relu(neg_sim.detach() - self.circle_margin)
        # alpha_noise=F.relu(1 -noise_sim.detach())
        #
        # logit_p = -self.circle_gamma * alpha_p * (pos_sim - 1+self.circle_margin)
        # logit_n = self.circle_gamma * alpha_n * alpha_noise * (neg_sim - self.circle_margin)
        #
        #
        # log_pos = torch.logsumexp(logit_p, dim=0, keepdim=True)  # [batch_size, 1]
        # log_neg = torch.logsumexp(logit_n, dim=0, keepdim=True)  # [batch_size, 1]
        #
        # #loss = torch.log(1 + torch.sum(torch.exp(logit_n)) * torch.sum(torch.exp(logit_p)))
        # soft_plus = nn.Softplus()
        # loss = soft_plus(log_pos + log_neg)
        #
        # # loss = torch.log(1 + torch.sum(torch.exp(logit_n)) * torch.sum(torch.exp(logit_p)))
        #
        # return loss.mean()

class CircleLoss_inter(nn.Module):
    """
    Circle Loss: A Unified Perspective of Pair Similarity Optimization
    https://arxiv.org/abs/2002.10857

    对称版本的Circle Loss，适用于对比学习场景
    """

    def __init__(self,
                 m: float = 0.25,
                 gamma: float = 256.0,  # 缩放因子，相当于1/τ
                 delta_p=0.75,
                 delta_n=0.25,
                 reduction: str = 'mean',
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        参数:
            m: 间隔参数，控制正负样本相似度的目标间隔
            gamma: 缩放因子，控制梯度幅度（类似于1/温度τ）
            delta_p: 正样本最优相似度目标 (1-m)
            delta_n: 负样本最优相似度目标 (m)
            reduction: 'mean'或'sum'，指定损失减少方式
            device: 计算设备
        """
        super().__init__()

        self.m = m
        self.gamma = gamma
        self.delta_p = delta_p
        self.delta_n = delta_n
        self.reduction = reduction
        self.device = device

        # 验证参数
        assert delta_p > delta_n, f"delta_p ({delta_p}) must be greater than delta_n ({delta_n})"
        assert reduction in ['mean', 'sum'], "reduction must be 'mean' or 'sum'"

    def forward(self, image_features1, image_features2, logit_scale):
        """
        前向传播

        参数:
            image_features1: 第一组特征 [batch_size, feature_dim]
            image_features2: 第二组特征 [batch_size, feature_dim]
            logit_scale: 可学习的缩放参数，通常用于控制相似度范围

        返回:
            对称Circle Loss
        """
        # 归一化特征向量
        image_features1 = F.normalize(image_features1, dim=-1)
        image_features2 = F.normalize(image_features2, dim=-1)

        # 计算相似度矩阵
        # logits_per_image1[i, j] = similarity(image1[i], image2[j])
        logits_per_image1 = (image_features1 @ image_features2.T)

        # 对称版本：交换顺序
        logits_per_image2 = logits_per_image1.T

        logits_image1_1 = (image_features1 @ image_features1.T)

        logits_image2_1 = (image_features2 @ image_features2.T)

        # labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=self.device)
        # 计算对称损失
        loss1 = self._circle_loss_single(logits_per_image1, logit_scale)
        loss2 = self._circle_loss_single(logits_per_image2, logit_scale)

        loss1_1 = self._circle_loss_single(logits_image1_1, logit_scale)
        loss2_1 = self._circle_loss_single(logits_image2_1, logit_scale)

        # 对称损失平均
        loss = (loss1 + loss2) / 2 * 0.8 + (loss1_1 + loss2_1) / 2 * 0.2

        return loss

    def _circle_loss_single(self, similarity_matrix, logit_scale):
        """
        计算单方向的Circle Loss

        参数:
            similarity_matrix: 相似度矩阵 [batch_size, batch_size]
                              diagonal[i, i] 是正样本相似度
                              off-diagonal[i, j (j≠i)] 是负样本相似度

        返回:
            circle loss
        """
        batch_size = similarity_matrix.size(0)

        # 创建掩码：对角线为正样本，非对角线为负样本

        pos_mask = torch.eye(batch_size, dtype=torch.bool, device=self.device)
        neg_mask = ~pos_mask

        # 正样本相似度 (s_p)
        pos_similarities = similarity_matrix[pos_mask].unsqueeze(1)  # [batch_size, 1]

        # 负样本相似度 (s_n)
        neg_similarities = similarity_matrix[neg_mask].view(batch_size, batch_size - 1)  # [batch_size, batch_size-1]

        # 计算自适应权重 alpha_p 和 alpha_n
        # alpha_p = relu(O_p - s_p), 其中 O_p = delta_p
        # alpha_n = relu(s_n - O_n), 其中 O_n = delta_n
        alpha_p = torch.relu(self.delta_p - pos_similarities.detach())
        alpha_n = torch.relu(neg_similarities.detach() - self.delta_n - alpha_p.detach().mean())

        # 计算正样本项的logsumexp
        # log(1 + sum(exp(-gamma * alpha_p * (s_p - delta_p))))
        pos_term = -logit_scale * alpha_p * (pos_similarities - self.delta_p)
        log_pos = torch.logsumexp(pos_term, dim=1, keepdim=True)  # [batch_size, 1]

        # 计算负样本项的logsumexp
        # log(1 + sum(exp(gamma * alpha_n * (s_n - delta_n))))
        neg_term = logit_scale * alpha_n * (neg_similarities - self.delta_n)
        log_neg = torch.logsumexp(neg_term, dim=1, keepdim=True)  # [batch_size, 1]

        # Circle Loss公式: log(1 + exp(...)) 的稳定计算
        # 使用 logsumexp 技巧避免数值溢出
        # loss_per_sample = torch.log1p(torch.exp(log_neg + log_pos))
        soft_plus = nn.Softplus()
        loss_per_sample = soft_plus(log_neg + log_pos)
        # 根据reduction参数聚合损失
        if self.reduction == 'mean':
            return loss_per_sample.mean()
        elif self.reduction == 'sum':
            return loss_per_sample.sum()
        else:
            return loss_per_sample


class CircleLossWithSmoothing(nn.Module):
    def __init__(self, m=0.25, gamma=256.0, delta_p=0.75, delta_n=0.25, label_smoothing=0.0, device='cuda'):
        super().__init__()
        self.circle_loss = CircleLoss(m, gamma, delta_p, delta_n, device=device)
        self.label_smoothing = label_smoothing

    def forward(self, image_features1, image_features2, logit_scale=1.0):
        # 计算标准的Circle Loss
        loss = self.circle_loss(image_features1, image_features2, logit_scale)

        # 如果有标签平滑，可以添加额外的正则化项
        if self.label_smoothing > 0:
            # 计算相似度矩阵
            sim_matrix = logit_scale * (F.normalize(image_features1, dim=-1) @
                                        F.normalize(image_features2, dim=-1).T)

            # 标签平滑可以作为熵正则化
            probs = F.softmax(sim_matrix, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()

            # 鼓励适度的不确定性
            loss = loss + self.label_smoothing * entropy

        return loss
