from operator import concat

import torch
import torch.nn as nn
import torchvision
import math
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torchvision.models._utils import IntermediateLayerGetter
import timm
import numpy as np

# sys.path.append("/mnt2/wangyuntao/geoLoc/model")
import pywt

# sys.path.append("/mnt2/wangyuntao/geoLoc/model")
from .common import PositionEncodingSine,DualStreamHybridTransformer3_correct_more_dropout, DualStreamHybridTransformer3_correct,DualStreamHybridTransformer3_correct_new,scTransformerEncoder,Spatial_fusion_guide_TransformerLayer

# sys.path.append("..")
# from visual import gem

"""
ConvNext model name
{
    'convnext_tiny_in22ft1k': 'convnext_tiny.fb_in22k_ft_in1k',
    'convnext_small_in22ft1k': 'convnext_small.fb_in22k_ft_in1k',
    'convnext_base_in22ft1k': 'convnext_base.fb_in22k_ft_in1k',
    'convnext_large_in22ft1k': 'convnext_large.fb_in22k_ft_in1k',
    'convnext_xlarge_in22ft1k': 'convnext_xlarge.fb_in22k_ft_in1k',
    'convnext_tiny_384_in22ft1k': 'convnext_tiny.fb_in22k_ft_in1k_384',
    'convnext_small_384_in22ft1k': 'convnext_small.fb_in22k_ft_in1k_384',
    'convnext_base_384_in22ft1k': 'convnext_base.fb_in22k_ft_in1k_384',
    'convnext_large_384_in22ft1k': 'convnext_large.fb_in22k_ft_in1k_384',
    'convnext_xlarge_384_in22ft1k': 'convnext_xlarge.fb_in22k_ft_in1k_384',
    'convnext_tiny_in22k': 'convnext_tiny.fb_in22k',
    'convnext_small_in22k': 'convnext_small.fb_in22k',
    'convnext_base_in22k': 'convnext_base.fb_in22k',
    'convnext_large_in22k': 'convnext_large.fb_in22k',
    'convnext_xlarge_in22k': 'convnext_xlarge.fb_in22k',

    'convnextv2_tiny_22k_224_ema': 'convnextv2_tiny.fcmae_ft_in22k_in1k'
    'convnextv2_tiny_22k_384_ema': 'convnextv2_tiny.fcmae_ft_in22k_in1k_384'
}

"""


# class Backbone(nn.Module):
#     def __init__(self, model_name, bk_checkpoint, return_interm_layers: bool, img_size=[122, 671]):
#         super().__init__()
#         self.name = model_name
#         # print('\nname\n',  name)
#         if 'resnet' in self.name.lower():
#             backbone = getattr(torchvision.models, self.name.lower())(
#                 weights='{}_Weights.IMAGENET1K_V1'.format(self.name))
#             assert self.name in ('ResNet18', 'ResNet34', 'ResNet50'), "number of channels are hard coded"
#
#             if return_interm_layers:
#                 # return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
#                 return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
#                 self.strides = [8, 16, 32]
#                 if self.name == 'ResNet50':
#                     self.num_channels = [512, 1024, 2048]
#                 else:  # resnet18 / resnet34
#                     self.num_channels = [128, 256, 512]
#             else:
#                 return_layers = {'layer4': "0"}
#                 self.strides = [32]
#                 if self.name == 'ResNet50':
#                     self.num_channels = [2048]
#                 else:  # resnet18 / resnet34
#                     self.num_channels = [512]
#             self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
#             self.data_config = None
#         elif 'convnext' in self.name.lower():
#             self.backbone = timm.create_model(self.name, pretrained=True, num_classes=0,
#                                               pretrained_cfg_overlay=dict(file=bk_checkpoint))
#             self.data_config = timm.data.resolve_model_data_config(self.backbone)
#             if return_interm_layers:
#                 self.strides = [8, 16, 32]
#                 if 'base' in self.name.lower():
#                     self.num_channels = [256, 512, 1024]
#                 elif 'tiny' in self.name.lower():
#                     self.num_channels = [192, 384, 768]
#             else:
#                 self.strides = [32]
#                 self.num_channels = [1024]
#
#         else:
#             raise RuntimeError(f'error model_name [resnet* or convnext]')
#
#     def forward(self, x):
#         if 'resnet' in self.name.lower():
#             xs = self.backbone(x)
#             out = []
#             for _, x in xs.items():
#                 out.append(x)
#         if 'convnext' in self.name.lower():
#             x = self.backbone.stem(x)
#             x0 = self.backbone.stages[0](x)
#
#             x1 = self.backbone.stages[1](x0)
#             x2 = self.backbone.stages[2](x1)
#             x3 = self.backbone.stages[3](x2)
#             # print(x.shape, x0.shape, x1.shape, x2.shape, x3.shape)
#
#             out = [x1, x2, x3]
#         return out

class Backbone(nn.Module):
    def __init__(self, model_name, bk_checkpoint, return_interm_layers: bool, img_size):
        super().__init__()
        self.name = model_name
        if 'resnet' in self.name.lower():
            backbone = getattr(torchvision.models, self.name.lower())(
                weights='{}_Weights.IMAGENET1K_V1'.format(self.name))
            assert self.name in ('ResNet18', 'ResNet34', 'ResNet50'), "number of channels are hard coded"

            if return_interm_layers:
                # return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
                return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
                self.strides = [8, 16, 32]
                if self.name == 'ResNet50':
                    self.num_channels = [512, 1024, 2048]
                else:  # resnet18 / resnet34
                    self.num_channels = [128, 256, 512]
            else:
                return_layers = {'layer4': "0"}
                self.strides = [32]
                if self.name == 'ResNet50':
                    self.num_channels = [2048]
                else:  # resnet18 / resnet34
                    self.num_channels = [512]
            self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
            self.data_config = None
        elif 'convnext' in self.name.lower():
            self.backbone = timm.create_model(self.name, pretrained=True, num_classes=0,
                                              pretrained_cfg_overlay=dict(file=bk_checkpoint))
            self.data_config = timm.data.resolve_model_data_config(self.backbone)
            if return_interm_layers:
                self.strides = [8, 16, 32]
                if 'base' in self.name.lower():
                    self.num_channels = [256, 512, 1024]
                elif 'tiny' in self.name.lower():
                    self.num_channels = [192, 384, 768]
            else:
                self.strides = [32]
                self.num_channels = [1024]
        elif 'swin' in self.name.lower():
            if 'swinv2_base_patch4_window12to24_192to384_22kto1k_ft' in self.name:
                timm_model_name = 'swinv2_base_patch4_window12to24_192to384.ms_in22k_ft_in1k.pth'
            else:
                timm_model_name = self.name.lower()

                # 创建模型 - 注意这里不设置 pretrained，因为我们要从检查点加载
            # self.backbone = timm.create_model(
            #     timm_model_name,
            #     pretrained=True,
            #     num_classes=0,
            #     # features_only=True,
            #     pretrained_cfg_overlay=dict(file=bk_checkpoint),
            #     imagesize=img_size
            # )
            self.backbone = timm.create_model(
                timm_model_name,
                pretrained=True,
                num_classes=0,
                features_only=True,  # 关键：启用 features_only 模式
                out_indices=(1, 2, 3),  # 返回第1,2,3阶段的特征（对应下采样率为8,16,32）
                pretrained_cfg_overlay=dict(file=bk_checkpoint),
                img_size=img_size,
                # window_size=16,
            )

            if return_interm_layers:
                self.strides = [8, 16, 32]
                if 'base' in self.name.lower():
                    self.num_channels = [256, 512, 1024]
                elif 'tiny' in self.name.lower():
                    self.num_channels = [192, 384, 768]
            else:
                self.strides = [32]
                self.num_channels = [1024]

            # 获取特征信息
            # feature_info = self.backbone.feature_info
            # self.strides = [info['reduction'] for info in feature_info]
            # self.num_channels = [info['num_chs'] for info in feature_info]
            self.data_config = timm.data.resolve_model_data_config(self.backbone)

            if return_interm_layers:
                # Swin Transformer V2 通常有4个阶段，我们取后3个阶段
                # 对应索引为 0,1,2,3，我们取 1,2,3
                self.strides = [8, 16, 32]
                # Swin V2 Base 的特征维度
                self.num_channels = [256, 512, 1024]
                # 设置要提取的层
                self.out_indices = (1, 2, 3)
                # 更新模型以返回中间层
                self.backbone.output_norm = None  # 移除最后的norm层
                # self.backbone.forward_features = self._forward_features_interm
            else:
                self.strides = [32]
                self.num_channels = [1024]
                self.out_indices = (3,)
                self.backbone.output_norm = None
                # self.backbone.forward_features = self._forward_features_last

        else:
            raise RuntimeError(f'error model_name [resnet*, convnext*, or swin*]')


    def forward(self, x):
        if 'swin' in self.name.lower():
            # 调用修改后的前向传播
            out = self.backbone(x)
            # 如果返回的是单个张量而不是列表，则包装成列表
            if not isinstance(out, list):
                out = [out]
        converted_out = []
        for feat in out:
            # 检查当前维度格式
            if feat.dim() == 4:
                # 如果是 (B, H, W, C) 格式
                if feat.shape[-1] in self.num_channels or feat.shape[1] not in self.num_channels:
                    # 假设通道在最后一维，转换为 (B, C, H, W)
                    feat = feat.permute(0, 3, 1, 2)
            converted_out.append(feat)
        return converted_out

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


class BackboneEmbed1(nn.Module):
    def __init__(self, backbone_strides, backbone_num_channels, return_interm_layers: bool,
                 no_extra_downsample=False, wavelet='haar'):
        super().__init__()
        self.return_interm_layers = return_interm_layers
        self.no_extra_downsample = no_extra_downsample
        self.wavelet = wavelet

        # ConvNeXt Base的通道数: [256, 512, 1024]
        # 计算每个层对应的1/4通道数: [64, 128, 256]  进行了dwt就乘以4
        self.output_channels = [256, 256, 256]

        # # 使用第一个层的输出通道数作为位置编码维度
        # self.pos_embed = PositionEncodingSine(d_model=self.output_channels[0])
        # 为每一层创建独立的位置编码（不立即初始化，在forward中动态创建）
        self.pos_embeds = nn.ModuleList()
        for i in range(len(self.output_channels)):
            pos_embed = PositionEncodingSine(
                d_model=self.output_channels[i],
            )
            self.pos_embeds.append(pos_embed)

        # 小波下采样层配置
        # 第一层: 两次小波下采样 (4倍下采样)
        # 第二层: 一次小波下采样 (2倍下采样)
        # 第三层: 无小波下采样
        self.dwt_layers = nn.ModuleList()
        for i, in_channels in enumerate(backbone_num_channels):
            out_channels = self.output_channels[i]
            if i == 0:  # 第一层 - 两次小波下采样
                self.dwt_layers.append(nn.Sequential(
                    nn.Conv2d(in_channels, in_channels // 2, 1),  # 通道调整
                    nn.GroupNorm(32, in_channels // 2),
                    GenericDWT_2D(in_channels // 2, wavelet),
                    nn.Conv2d(2 * in_channels, in_channels, 1),  # 通道调整
                    nn.GroupNorm(32, in_channels),
                    GenericDWT_2D(in_channels, wavelet),
                    nn.Conv2d(in_channels * 4, out_channels, 1),  # 通道调整
                    nn.GroupNorm(32, out_channels),
                ))
            elif i == 1:  # 第二层 - 一次小波下采样
                self.dwt_layers.append(nn.Sequential(
                    nn.Conv2d(in_channels, in_channels // 2, 1),  # 通道调整
                    nn.GroupNorm(32, in_channels // 2),
                    GenericDWT_2D(in_channels // 2, wavelet),
                    nn.Conv2d(in_channels * 2, out_channels, 1),  # 通道调整
                    nn.GroupNorm(32, out_channels)
                ))
            else:  # 第三层 - 无小波下采样，仅通道调整
                self.dwt_layers.append(nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.GroupNorm(32, out_channels)
                ))

        # 额外下采样层（如果需要）
        if self.return_interm_layers and not self.no_extra_downsample:
            in_channels = backbone_num_channels[-1]  # 1024
            out_channels = in_channels // 4  # 256
            self.extra_downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(32, out_channels)
            )

    def forward(self, features):
        """
        Args:
            features: 主干网络输出的多尺度特征列表 [f1, f2, f3]
                    f1: [B, 256, H1, W1]
                    f2: [B, 512, H2, W2]
                    f3: [B, 1024, H3, W3]
        Returns:
            feats_embed: 经过小波下采样和位置编码的特征列表
            srcs: 原始投影后的特征列表
        """
        feats_embed = []
        srcs = []

        # 处理主干网络的所有特征层
        for l, feat in enumerate(features):
            # 应用小波下采样和通道调整
            src = self.dwt_layers[l](feat)
            srcs.append(src)

            # 动态创建或获取该层的位置编码
            p = self.pos_embeds[l](src)
            feats_embed.append(p)

        # 只有在需要额外下采样时才添加最后一层
        if self.return_interm_layers and not self.no_extra_downsample:
            src = self.extra_downsample(features[-1])
            srcs.append(src)
            p = self.pos_embed(src)
            feats_embed.append(p)

        return feats_embed

    def _get_pos_embed_for_layer(self, layer_idx, feat):
        """为特定层动态创建位置编码"""
        batch_size, channels, height, width = feat.shape

        # 如果该层的位置编码尚未创建或尺寸不匹配，则重新创建
        if (self.pos_embed_layers[layer_idx] is None or
                self.pos_embed_layers[layer_idx].shape[2] != height or
                self.pos_embed_layers[layer_idx].shape[3] != width):
            # 动态创建位置编码
            pos_embed = PositionEncodingSine(
                d_model=channels,
                max_shape=(height, width),
                temp_bug_fix=True
            )

            # 注册为缓冲区以便设备转移
            self.register_buffer(f'pos_embed_{layer_idx}', pos_embed.pe)
            self.pos_embed_layers[layer_idx] = pos_embed.pe

        return self.pos_embed_layers[layer_idx]

class BackboneEmbed1Conv(nn.Module):
    def __init__(self, backbone_strides, backbone_num_channels, return_interm_layers: bool,
                 no_extra_downsample=False, wavelet='haar'):  # wavelet参数保留但不再使用
        super().__init__()
        self.return_interm_layers = return_interm_layers
        self.no_extra_downsample = no_extra_downsample
        # wavelet参数不再使用，但保留接口兼容性

        # 最终每层输出通道数固定为256
        self.output_channels = [256, 256, 256]

        # 为每一层创建独立的位置编码
        self.pos_embeds = nn.ModuleList()
        for i in range(len(self.output_channels)):
            pos_embed = PositionEncodingSine(d_model=self.output_channels[i])
            self.pos_embeds.append(pos_embed)

        # 用卷积下采样层替代原来的小波层
        self.conv_down_layers = nn.ModuleList()
        for i, in_channels in enumerate(backbone_num_channels):
            out_channels = self.output_channels[i]
            if i == 0:  # 第一层：两次下采样（4倍下采样）
                self.conv_down_layers.append(nn.Sequential(
                    # 第一次下采样：stride=2卷积，通道减半
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, in_channels ),
                    # 第二次下采样：stride=2卷积，输出目标通道
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, out_channels)
                ))
            elif i == 1:  # 第二层：一次下采样（2倍下采样）
                self.conv_down_layers.append(nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, out_channels)
                ))
            else:  # 第三层：无下采样，仅通道调整
                self.conv_down_layers.append(nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1),
                    nn.GroupNorm(32, out_channels)
                ))

        # 额外下采样层（如果需要）
        if self.return_interm_layers and not self.no_extra_downsample:
            in_channels = backbone_num_channels[-1]  # 最后一层输入通道
            out_channels = in_channels // 4          # 目标通道数（通常256）
            self.extra_downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(32, out_channels)
            )

    def forward(self, features):
        """
        Args:
            features: 主干网络输出的多尺度特征列表 [f1, f2, f3]
                    f1: [B, C1, H1, W1]
                    f2: [B, C2, H2, W2]
                    f3: [B, C3, H3, W3]
        Returns:
            feats_embed: 经过卷积下采样和位置编码的特征列表
            srcs: 原始投影后的特征列表
        """
        feats_embed = []
        srcs = []

        # 处理主干网络的所有特征层
        for l, feat in enumerate(features):
            # 应用卷积下采样和通道调整
            src = self.conv_down_layers[l](feat)
            srcs.append(src)

            # 添加位置编码
            p = self.pos_embeds[l](src)
            feats_embed.append(p)

        # 如果需要额外下采样，对最后一层特征再下采样一次
        if self.return_interm_layers and not self.no_extra_downsample:
            src = self.extra_downsample(features[-1])
            srcs.append(src)
            # 注意：这里原来使用了 self.pos_embed，但已经替换为列表，应使用新的pos_embed
            # 由于额外下采样层输出的通道数与第三层相同（256），我们可以复用第三层的位置编码
            p = self.pos_embeds[-1](src)  # 使用最后一层的位置编码
            feats_embed.append(p)

        return feats_embed

    def _get_pos_embed_for_layer(self, layer_idx, feat):
        """为特定层动态创建位置编码"""
        batch_size, channels, height, width = feat.shape

        # 如果该层的位置编码尚未创建或尺寸不匹配，则重新创建
        if (self.pos_embed_layers[layer_idx] is None or
                self.pos_embed_layers[layer_idx].shape[2] != height or
                self.pos_embed_layers[layer_idx].shape[3] != width):
            # 动态创建位置编码
            pos_embed = PositionEncodingSine(
                d_model=channels,
                max_shape=(height, width),
                temp_bug_fix=True
            )

            # 注册为缓冲区以便设备转移
            self.register_buffer(f'pos_embed_{layer_idx}', pos_embed.pe)
            self.pos_embed_layers[layer_idx] = pos_embed.pe

        return self.pos_embed_layers[layer_idx]

class GenericDWT_2D(nn.Module):
    """支持多种小波基的二维离散小波变换层"""

    def __init__(self, in_channels, wavelet):
        super().__init__()
        self.in_channels = in_channels
        self.wavelet = wavelet

        # 获取小波滤波器系数
        wavelet_obj = pywt.Wavelet(wavelet)
        dec_lo, dec_hi, rec_lo, rec_hi = wavelet_obj.filter_bank[:4]

        # 创建二维滤波器
        ll = np.outer(dec_lo, dec_lo)
        lh = np.outer(dec_lo, dec_hi)
        hl = np.outer(dec_hi, dec_lo)
        hh = np.outer(dec_hi, dec_hi)

        # 归一化
        filters = np.stack([ll, lh, hl, hh], axis=0)
        # filters = filters / np.max(np.abs(filters))

        # 注册为缓冲区
        filters = filters[:, np.newaxis, :, :]
        self.register_buffer('base_weight', torch.tensor(filters, dtype=torch.float32))

    def forward(self, x):
        # 动态扩展滤波器
        kernel_size = self.base_weight.shape[2]
        weight = self.base_weight.repeat(self.in_channels, 1, 1, 1)

        # 计算所需填充
        padding = (kernel_size - 2) // 2
        return F.conv2d(x, weight, stride=2, padding=padding, groups=self.in_channels)


class UnifiedSizeTransformer(nn.Module):
    """专门处理尺寸统一的模块"""

    def __init__(self, backbone_num_channels, target_size, wavelet='haar'):
        super().__init__()
        self.backbone_num_channels = backbone_num_channels
        self.target_size = target_size
        self.target_channelsize = [4 * backbone_num_channels[0], backbone_num_channels[1] * 2, backbone_num_channels[2]]
        self.wavelet = wavelet

        # 精确的下采样层
        self.unify_layers = nn.ModuleList()

        # 第一层：两次DWT下采样 (4倍下采样)
        self.unify_layers.append(nn.Sequential(
            GenericDWT_2D(backbone_num_channels[0], wavelet),
            nn.Conv2d(backbone_num_channels[0] * 4, 2 * backbone_num_channels[0], 1),
            nn.GroupNorm(32, 2 * backbone_num_channels[0]),
            GenericDWT_2D(2 * backbone_num_channels[0], wavelet),
            nn.Conv2d(backbone_num_channels[0] * 8, self.target_channelsize[0], 1),
            nn.GroupNorm(32, self.target_channelsize[0]),
        ))

        # 第二层：一次DWT下采样 (2倍下采样)
        self.unify_layers.append(nn.Sequential(
            GenericDWT_2D(backbone_num_channels[1], wavelet),
            nn.Conv2d(backbone_num_channels[1] * 4, self.target_channelsize[1], 1),
            nn.GroupNorm(32, self.target_channelsize[1]),
        ))

        # 第三层：Identity (保持原样)
        self.unify_layers.append(nn.Identity())

    def forward(self, features_list):
        unified_features = []
        for feat, unify_layer in zip(features_list, self.unify_layers):
            unified_feat = unify_layer(feat)
            unified_features.append(unified_feat)
        return unified_features


class GenericIDWT_2D(nn.Module):
    """支持多种小波基的二维小波逆变换层"""

    def __init__(self, in_channels, wavelet):
        super().__init__()
        if in_channels % 4 != 0:
            raise ValueError(f"Input channels must be multiple of 4, got {in_channels}")

        self.wavelet = wavelet
        self.in_channels = in_channels
        self.groups = in_channels // 4

        # 获取小波滤波器系数
        wavelet_obj = pywt.Wavelet(wavelet)
        dec_lo, dec_hi, rec_lo, rec_hi = wavelet_obj.filter_bank[:4]

        # 创建二维重构滤波器
        ll = np.outer(rec_lo, rec_lo)
        lh = np.outer(rec_lo, rec_hi)
        hl = np.outer(rec_hi, rec_lo)
        hh = np.outer(rec_hi, rec_hi)

        # 归一化
        filters = np.stack([ll, lh, hl, hh], axis=0)
        # filters = filters / np.max(np.abs(filters))

        # 注册为缓冲区
        filters = filters[:, np.newaxis, :, :]
        self.register_buffer('base_weight', torch.tensor(filters, dtype=torch.float32))

    def forward(self, x):
        kernel_size = self.base_weight.shape[2]
        weight = self.base_weight.repeat(self.groups, 1, 1, 1)

        # 计算输出填充
        padding = (kernel_size - 2) // 2
        return F.conv_transpose2d(x, weight, stride=2, padding=padding, groups=self.groups)


# 输出的结果为256 256 256 12*12
# 输出的结果为64 48 48    128 24 24  256 12*12
class UnifiedSizeCNN(nn.Module):
    """专门处理尺寸统一的模块"""

    def __init__(self, backbone_num_channels, wavelet='haar'):
        super().__init__()
        self.backbone_num_channels = backbone_num_channels
        self.target_channelsize = [backbone_num_channels[0] // 4, backbone_num_channels[1] // 2,
                                   backbone_num_channels[2]]
        self.wavelet = wavelet

        # 精确的下采样层
        self.unify_layers = nn.ModuleList()

        # 第一层：两次DWT下采样 (4倍下采样)
        self.unify_layers.append(nn.Sequential(
            GenericIDWT_2D(backbone_num_channels[0], wavelet),
            nn.Conv2d(backbone_num_channels[0] // 4, backbone_num_channels[0] // 2, 1),
            nn.GroupNorm(32, backbone_num_channels[0] // 2),
            GenericIDWT_2D(backbone_num_channels[0] // 2, wavelet),
            nn.Conv2d(backbone_num_channels[0] // 8, self.target_channelsize[0], 1),
            nn.GroupNorm(32, self.target_channelsize[0]),
        ))

        # 第二层：一次DWT下采样 (2倍下采样)
        self.unify_layers.append(nn.Sequential(
            GenericIDWT_2D(backbone_num_channels[1], wavelet),
            nn.Conv2d(backbone_num_channels[1] // 4, self.target_channelsize[1], 1),
            nn.GroupNorm(32, self.target_channelsize[1]),
        ))

        # 第三层：Identity (保持原样)
        self.unify_layers.append(nn.Identity())

    def forward(self, features_list):
        unified_features = []
        for feat, unify_layer in zip(features_list, self.unify_layers):
            unified_feat = unify_layer(feat)
            unified_features.append(unified_feat)
        return unified_features


## for Univerity-1652, branch sat and grd share weights
class TimmModel_u(nn.Module):
    def __init__(self, model_name,
                 img_size,
                 psm=True,
                 is_polar=False,
                 no_extra_downsample=True):

        super(TimmModel_u, self).__init__()

        self.is_polar = is_polar
        self.backbone_name = model_name
        self.img_size = (img_size, img_size)
        self.no_extra_downsample = no_extra_downsample  # 保存参数

        self.d_model = 128
        self.nheads = 4
        self.nlayers = 2
        self.ffn_dim = 1024
        self.dropout = 0.3
        self.em_dim = 2048

        self.activation = nn.GELU()
        self.single_features = False

        self.sample = psm

        if 'tiny' in self.backbone_name:
            if 'v2' in self.backbone_name:
                self.bk_checkpoint = 'pretrained/convnextv2_tiny_22k_224_ema.pt'  # '/mnt2/wangyuntao/pretrained/convnextv2_tiny_22k_384_ema.pt'
            else:
                self.bk_checkpoint = 'pretrained/convnext_tiny_22k_1k_224.pth'
        elif 'base' in self.backbone_name:
        #     if 'v2' in self.backbone_name:
        #         self.bk_checkpoint = 'pretrained/convnextv2_base_22k_224_ema.pt'
        #     else:
        #         self.bk_checkpoint = 'pretrained/convnext_base_22k_1k_224.pth'
        # elif 'base' in self.backbone_name:
            if 'v2' in self.backbone_name:
                self.bk_checkpoint = 'pretrained/swinv2_base_patch4_window12to24_192to384_22kto1k_ft.pth'
            else:
                self.bk_checkpoint = 'pretrained/swinv2_base_patch4_window12to24_192to384_22kto1k_ft.pth'
        else:
            self.bk_checkpoint = None

        # if '384' in self.backbone_name:
        #     self.bk_checkpoint = self.bk_checkpoint.replace('224', '384')

        # Backbone
        self.backbone = Backbone(self.backbone_name, self.bk_checkpoint, return_interm_layers=not self.single_features, img_size=self.img_size)

        # self.embed1 = BackboneEmbed1(self.backbone.strides, self.backbone.num_channels,
        #                              return_interm_layers=not self.single_features,
        #                              no_extra_downsample=self.no_extra_downsample, wavelet='haar')

        self.embed1 = BackboneEmbed1Conv(self.backbone.strides, self.backbone.num_channels,
                                     return_interm_layers=not self.single_features,
                                     no_extra_downsample=self.no_extra_downsample, wavelet='haar')

        self.spatial_transformer = DualStreamHybridTransformer3_correct(
            output_channels=self.embed1.output_channels,  # [256, 256, 256]
            spatial_size=int(img_size / self.backbone.strides[2]),
            ffn_dim_ratio=2,
            dropout=self.dropout
        )
        # 计算目标尺寸（第三层的尺寸）
        strides = self.backbone.strides  # [8, 16, 32]
        self.target_size = (math.floor(img_size / strides[2]), math.floor(img_size / strides[2]))


        # self.feat_dim, self.H, self.W = self._dim(self.backbone_name, self.backbone.strides,
        #                                           img_size=self.img_size)

        #
        self.num_channles1 = self.backbone.num_channels  # [256, 512, 1025] local

        out_dim_g1 = 4  # 256*8
        gobal_in_dim1 = int(3 * img_size / strides[2] * img_size / strides[2])

        self.proj1 = nn.Linear(gobal_in_dim1, out_dim_g1)

        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.sigmoid = nn.Sigmoid()

        self.pool_h = nn.AdaptiveAvgPool2d((1, None))  # 输出形状: (B, C, 1, W)   %AdaptiveMaxPool2d
        # 对W维度池化 (保持H维度不变)
        self.pool_w = nn.AdaptiveAvgPool2d((None, 1))  # 输出形状: (B, C, H, 1)

        self.unify_size_cnn = UnifiedSizeCNN(
            backbone_num_channels=self.embed1.output_channels,
        )
        self.num_channles = self.unify_size_cnn.target_channelsize  # [64, 128, 256]全局


        self.sigmoid = nn.Sigmoid()

        # 在 __init__ 尾部添加，替代硬编码的 global_scale
        self.global_scale_logit = nn.Parameter(torch.tensor(-1.51))  # softplus(-1.6) ≈ 0.2


    def get_config(self, ):
        data_config = self.backbone.data_config
        return data_config

    def set_grad_checkpointing(self, enable=True):
        self.model.set_grad_checkpointing(enable)

    def k_size(self, in_dim):
        t = int(abs((math.log(in_dim, 2) + 1) / 2))
        k_size = t if t % 2 else t + 1

        return k_size

    # @autocast()
    def forward(self, img1, img2=None, input_id=1):
        if img2 is not None:
            grd_b = img1.shape[0]
            sat_b = img2.shape[0]
            sat_x = self.backbone(img2)
            grd_x = self.backbone(img1)
            ## global
            sat_e1 = self.embed1(sat_x)
            grd_e1 = self.embed1(grd_x)

            # 应用SimplifiedCrossScaleTransformer
            sat_fusion = self.spatial_transformer(sat_e1)
            grd_fusion = self.spatial_transformer(grd_e1)

            sat_global1 = torch.cat([sat_fusion[0].flatten(2), sat_fusion[1].flatten(2), sat_fusion[2].flatten(2)],
                                    dim=2)
            grd_global1 = torch.cat([grd_fusion[0].flatten(2), grd_fusion[1].flatten(2), grd_fusion[2].flatten(2)],
                                    dim=2)
            sat_global1 = self.proj1(sat_global1.flatten(2)).contiguous().view(sat_b, -1)
            grd_global1 = self.proj1(grd_global1.flatten(2)).contiguous().view(grd_b, -1)


            sat_local_spatial2=self.avg_pool(sat_x[2]).squeeze(2).squeeze(-1)
            grd_local_spatial2=self.avg_pool(grd_x[2]).squeeze(2).squeeze(-1)

            sat_global1=F.normalize(sat_global1.contiguous(), p=2, dim=1)
            grd_global1=F.normalize(grd_global1.contiguous(), p=2, dim=1)
            sat_local_spatial2=F.normalize(sat_local_spatial2.contiguous(), p=2, dim=1)
            grd_local_spatial2=F.normalize(grd_local_spatial2.contiguous(), p=2, dim=1)

            # 降低 global 的影响系数
            global_scale = F.softplus(self.global_scale_logit)  # 正值，可微

            desc_sat = torch.cat([global_scale * sat_global1, sat_local_spatial2], dim=1)
            desc_grd = torch.cat([global_scale * grd_global1, grd_local_spatial2], dim=1)

            desc_sat = F.normalize(desc_sat.contiguous(), p=2, dim=1)
            desc_grd = F.normalize(desc_grd.contiguous(), p=2, dim=1)

            return desc_sat.contiguous(), desc_grd.contiguous()

        else:
            b, _, h, w = img1.shape

            sat_x = self.backbone(img1)
            # 新增的单分支处理部分
            sat_e1 = self.embed1(sat_x)

            sat_fusion = self.spatial_transformer(sat_e1)

            sat_global1 = torch.cat([sat_fusion[0].flatten(2), sat_fusion[1].flatten(2), sat_fusion[2].flatten(2)],
                                    dim=2)
            sat_global1 = self.proj1(sat_global1.flatten(2)).contiguous().view(b, -1)

            sat_local_spatial2=self.avg_pool(sat_x[2]).squeeze(2).squeeze(-1)

            sat_global1=F.normalize(sat_global1.contiguous(), p=2, dim=1)
            sat_local_spatial2=F.normalize(sat_local_spatial2.contiguous(), p=2, dim=1)

            global_scale = F.softplus(self.global_scale_logit)  # 正值，可微
            desc_sat = torch.cat([global_scale * sat_global1, sat_local_spatial2], dim=1)

            desc_sat = F.normalize(desc_sat.contiguous(), p=2, dim=1)

            return desc_sat.contiguous()

    def _reshape_feat(self, feat_H, H, W):
        p1 = H[0] * W[0]
        p2 = H[-1] * W[-1]
        feat_h1 = feat_H[:, :p1, :].contiguous()
        feat_h2 = feat_H[:, p1:-p2, :].contiguous()
        feat_h3 = feat_H[:, -p2:, :].contiguous()

        return [feat_h1, feat_h2, feat_h3]

    def _dim(self, model_name, strides, img_size=[122, 671], no_extra_downsample=True):
        if 'convnext' in model_name.lower():
            H = [math.floor(img_size[0] / r) for r in strides]
            W = [math.floor(img_size[1] / r) for r in strides]
            feat_dim = [H[i] * W[i] for i in range(len(H))]
        elif 'resnet' in model_name.lower():
            H = [math.ceil(img_size[0] / r) for r in strides]
            W = [math.ceil(img_size[1] / r) for r in strides]
            feat_dim = [H[i] * W[i] for i in range(len(H))]

        # 只有在需要额外下采样时才添加最后一层
        if not no_extra_downsample:
            H.append(math.ceil(H[-1] / 2))
            W.append(math.ceil(W[-1] / 2))
            feat_dim.append(H[-1] * W[-1])

        return feat_dim, H, W

    def compute_sparsity_loss(self):
        """返回标量 L1 惩罚项，不乘系数，系数由外部控制"""
        return F.softplus(self.global_scale_logit)