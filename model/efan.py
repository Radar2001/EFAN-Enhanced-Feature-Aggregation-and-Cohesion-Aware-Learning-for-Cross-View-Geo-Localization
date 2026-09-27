"""Enhanced Feature Aggregation Network (EFAN) for cross-view geo-localization.

EFAN is a shared-weight siamese network that couples:

* the Scale-Unified Semantic-Spatial Feature Enhancement Module (S3EM), and
* the Complementary Feature Aggregation (CFA) strategy.

The CFA explicitly fuses the backbone's global semantic anchor (GAP of the
deepest backbone feature) with the spatially enhanced representation output by
the S3EM, yielding a robust yet discriminative image representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np

from .modules import PositionEncodingSine, S3EM


class Backbone(nn.Module):
    """Feature extractor supporting ConvNeXt / ConvNeXtV2 / Swin / SwinV2.

    Returns the multi-scale features of the last three stages (stride 8, 16, 32).
    """

    def __init__(self, model_name, bk_checkpoint, return_interm_layers: bool, img_size):
        super().__init__()
        self.name = model_name
        name = self.name.lower()

        if return_interm_layers:
            self.strides = [8, 16, 32]
            if 'base' in name:
                self.num_channels = [256, 512, 1024]
            elif 'tiny' in name:
                self.num_channels = [192, 384, 768]
        else:
            self.strides = [32]
            self.num_channels = [1024]

        if 'convnext' in name:
            self.backbone = timm.create_model(
                self.name,
                pretrained=True,
                num_classes=0,
                pretrained_cfg_overlay=dict(file=bk_checkpoint),
            )
            self.data_config = timm.data.resolve_model_data_config(self.backbone)
        elif 'swin' in name:
            if 'swinv2_base_patch4_window12to24_192to384_22kto1k_ft' in self.name:
                timm_model_name = 'swinv2_base_patch4_window12to24_192to384.ms_in22k_ft_in1k.pth'
            else:
                timm_model_name = self.name.lower()

            self.backbone = timm.create_model(
                timm_model_name,
                pretrained=True,
                num_classes=0,
                features_only=True,  # return multi-scale features
                out_indices=(1, 2, 3),
                pretrained_cfg_overlay=dict(file=bk_checkpoint),
                img_size=img_size,
            )
            self.data_config = timm.data.resolve_model_data_config(self.backbone)

            if return_interm_layers:
                self.out_indices = (1, 2, 3)
                self.backbone.output_norm = None
            else:
                self.out_indices = (3,)
                self.backbone.output_norm = None
        else:
            raise RuntimeError(f'Unsupported backbone: {self.name}')

    def forward(self, x):
        if 'convnext' in self.name.lower():
            # ConvNeXt / ConvNeXtV2: manually take the last three stages
            x = self.backbone.stem(x)
            x0 = self.backbone.stages[0](x)
            x1 = self.backbone.stages[1](x0)
            x2 = self.backbone.stages[2](x1)
            x3 = self.backbone.stages[3](x2)
            out = [x1, x2, x3]
        elif 'swin' in self.name.lower():
            # Swin (V1/V2): features_only mode returns multi-scale features
            out = self.backbone(x)
            if not isinstance(out, list):
                out = [out]
        else:
            raise RuntimeError(f'Unsupported backbone: {self.name}')

        converted_out = []
        for feat in out:
            if feat.dim() == 4:
                # ensure the channel dimension is at index 1 (B, C, H, W)
                if feat.shape[-1] in self.num_channels or feat.shape[1] not in self.num_channels:
                    feat = feat.permute(0, 3, 1, 2)
            converted_out.append(feat)
        return converted_out


class ScaleUnification(nn.Module):
    """Unifies multi-scale backbone features to a common scale and channel count.

    S1 is downsampled twice (4x) and S2 once (2x) by stride-2 convolutions so
    that all features align with S3's resolution; all channels are standardized
    to 256. A sinusoidal position encoding is then added per level.
    """

    def __init__(self, backbone_strides, backbone_num_channels,
                 return_interm_layers: bool, no_extra_downsample=False):
        super().__init__()
        self.return_interm_layers = return_interm_layers
        self.no_extra_downsample = no_extra_downsample

        self.output_channels = [256, 256, 256]

        self.pos_embeds = nn.ModuleList([
            PositionEncodingSine(d_model=ch) for ch in self.output_channels
        ])

        self.conv_down_layers = nn.ModuleList()
        for i, in_channels in enumerate(backbone_num_channels):
            out_channels = self.output_channels[i]
            if i == 0:  # two stride-2 downsampling operations (4x)
                self.conv_down_layers.append(nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, in_channels),
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, out_channels),
                ))
            elif i == 1:  # one stride-2 downsampling operation (2x)
                self.conv_down_layers.append(nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, out_channels),
                ))
            else:  # only channel adjustment
                self.conv_down_layers.append(nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1),
                    nn.GroupNorm(32, out_channels),
                ))

        if self.return_interm_layers and not self.no_extra_downsample:
            in_channels = backbone_num_channels[-1]
            out_channels = in_channels // 4
            self.extra_downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(32, out_channels),
            )

    def forward(self, features):
        feats_embed = []

        for l, feat in enumerate(features):
            src = self.conv_down_layers[l](feat)
            feats_embed.append(self.pos_embeds[l](src))

        if self.return_interm_layers and not self.no_extra_downsample:
            src = self.extra_downsample(features[-1])
            feats_embed.append(self.pos_embeds[-1](src))

        return feats_embed


class EFAN(nn.Module):
    """Enhanced Feature Aggregation Network (EFAN).

    A shared-weight siamese network for cross-view geo-localization. It first
    unifies multi-scale backbone features and enhances them via the S3EM, then
    applies the CFA strategy: the global semantic anchor (GAP of the deepest
    backbone feature) is concatenated with the learnable-scaled spatial
    projection of the S3EM output, and the result is L2-normalized.
    """

    def __init__(self, model_name, img_size, no_extra_downsample=True):
        super().__init__()
        self.backbone_name = model_name
        self.img_size = (img_size, img_size)
        self.no_extra_downsample = no_extra_downsample

        self.dropout = 0.3
        self.single_features = False

        name = self.backbone_name.lower()
        if 'convnextv2' in name:
            if 'tiny' in name:
                self.bk_checkpoint = 'pretrained/convnextv2_tiny_22k_224_ema.pt'
            elif 'base' in name:
                self.bk_checkpoint = 'pretrained/convnextv2_base_22k_224_ema.pt'
            else:
                self.bk_checkpoint = None
        elif 'convnext' in name:
            if 'tiny' in name:
                self.bk_checkpoint = 'pretrained/convnext_tiny_22k_1k_224.pth'
            elif 'base' in name:
                self.bk_checkpoint = 'pretrained/convnext_base_22k_1k_224.pth'
            else:
                self.bk_checkpoint = None
        elif 'swin' in name:
            if 'v2' in name:
                if 'tiny' in name:
                    self.bk_checkpoint = 'pretrained/swinv2_tiny_patch4_window8_256.pth'
                else:
                    self.bk_checkpoint = 'pretrained/swinv2_base_patch4_window12to24_192to384_22kto1k_ft.pth'
            else:
                # Swin Transformer V1 (Base)
                self.bk_checkpoint = 'pretrained/swin_base_patch4_window7_224_22kto1k.pth'
        else:
            self.bk_checkpoint = None

        self.backbone = Backbone(
            self.backbone_name,
            self.bk_checkpoint,
            return_interm_layers=not self.single_features,
            img_size=self.img_size,
        )

        self.scale_unification = ScaleUnification(
            self.backbone.strides,
            self.backbone.num_channels,
            return_interm_layers=not self.single_features,
            no_extra_downsample=self.no_extra_downsample,
        )

        self.s3em = S3EM(
            output_channels=self.scale_unification.output_channels,
            spatial_size=int(img_size / self.backbone.strides[2]),
            ffn_dim_ratio=2,
            dropout=self.dropout,
        )

        strides = self.backbone.strides
        out_dim_g = 4
        global_in_dim = int(3 * img_size / strides[2] * img_size / strides[2])
        self.proj = nn.Linear(global_in_dim, out_dim_g)

        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Learnable scalar alpha controlling the spatial branch weight in CFA.
        self.global_scale_logit = nn.Parameter(torch.tensor(-1.51))

    def get_config(self):
        return self.backbone.data_config

    def forward(self, img1, img2=None, input_id=1):
        if img2 is not None:
            grd_b = img1.shape[0]
            sat_b = img2.shape[0]

            sat_feats = self.backbone(img2)
            grd_feats = self.backbone(img1)

            sat_unified = self.scale_unification(sat_feats)
            grd_unified = self.scale_unification(grd_feats)

            sat_s3em = self.s3em(sat_unified)
            grd_s3em = self.s3em(grd_unified)

            sat_global = torch.cat(
                [sat_s3em[0].flatten(2), sat_s3em[1].flatten(2), sat_s3em[2].flatten(2)], dim=2
            )
            grd_global = torch.cat(
                [grd_s3em[0].flatten(2), grd_s3em[1].flatten(2), grd_s3em[2].flatten(2)], dim=2
            )
            sat_global = self.proj(sat_global.flatten(2)).contiguous().view(sat_b, -1)
            grd_global = self.proj(grd_global.flatten(2)).contiguous().view(grd_b, -1)

            sat_local = self.avg_pool(sat_feats[2]).squeeze(2).squeeze(-1)
            grd_local = self.avg_pool(grd_feats[2]).squeeze(2).squeeze(-1)

            sat_global = F.normalize(sat_global.contiguous(), p=2, dim=1)
            grd_global = F.normalize(grd_global.contiguous(), p=2, dim=1)
            sat_local = F.normalize(sat_local.contiguous(), p=2, dim=1)
            grd_local = F.normalize(grd_local.contiguous(), p=2, dim=1)

            alpha = F.softplus(self.global_scale_logit)

            desc_sat = torch.cat([alpha * sat_global, sat_local], dim=1)
            desc_grd = torch.cat([alpha * grd_global, grd_local], dim=1)

            desc_sat = F.normalize(desc_sat.contiguous(), p=2, dim=1)
            desc_grd = F.normalize(desc_grd.contiguous(), p=2, dim=1)

            return desc_sat.contiguous(), desc_grd.contiguous()

        else:
            b, _, h, w = img1.shape

            sat_feats = self.backbone(img1)
            sat_unified = self.scale_unification(sat_feats)
            sat_s3em = self.s3em(sat_unified)

            sat_global = torch.cat(
                [sat_s3em[0].flatten(2), sat_s3em[1].flatten(2), sat_s3em[2].flatten(2)], dim=2
            )
            sat_global = self.proj(sat_global.flatten(2)).contiguous().view(b, -1)

            sat_local = self.avg_pool(sat_feats[2]).squeeze(2).squeeze(-1)

            sat_global = F.normalize(sat_global.contiguous(), p=2, dim=1)
            sat_local = F.normalize(sat_local.contiguous(), p=2, dim=1)

            alpha = F.softplus(self.global_scale_logit)
            desc_sat = torch.cat([alpha * sat_global, sat_local], dim=1)
            desc_sat = F.normalize(desc_sat.contiguous(), p=2, dim=1)

            return desc_sat.contiguous()
