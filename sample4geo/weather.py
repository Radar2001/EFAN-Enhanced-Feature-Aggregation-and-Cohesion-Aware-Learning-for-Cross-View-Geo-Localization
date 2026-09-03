# -*- coding: utf-8 -*-
"""University-1652 多天气增强（基于 albumentations）。

与原始 MuseNet 的 iaa_weather_list 保持一致，共 10 种天气条件：
normal, fog, rain, snow, dark, overexposure, fog+rain, fog+snow, rain+snow, wind。
"""

import random
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform

WEATHER_NAMES = [
    "normal",       # 0
    "fog",          # 1
    "rain",         # 2
    "snow",         # 3
    "dark",         # 4
    "overexposure", # 5
    "fog+rain",     # 6
    "fog+snow",     # 7
    "rain+snow",    # 8
    "wind",         # 9
]


def _fog():
    return A.RandomFog(fog_coef_lower=0.5, fog_coef_upper=0.8, alpha_coef=0.08, p=1.0)


def _rain():
    return A.RandomRain(slant_lower=-10, slant_upper=10,
                        drop_length=10, drop_width=1,
                        drop_color=(200, 200, 200),
                        blur_value=3, brightness_coefficient=0.8,
                        rain_type="drizzle", p=1.0)


def _snow():
    return A.RandomSnow(snow_point_lower=0.2, snow_point_upper=0.4,
                        brightness_coeff=2.0, p=1.0)


def get_weather_ops():
    """返回 10 个天气操作（索引 0 为 None，即 normal 不做增强）。"""
    return [
        None,                          # normal
        _fog(),                        # fog
        _rain(),                       # rain
        _snow(),                       # snow
        A.RandomBrightnessContrast(brightness_limit=(-0.5, -0.3),
                                   contrast_limit=(-0.2, 0.0), p=1.0),  # dark
        A.RandomBrightnessContrast(brightness_limit=(0.4, 0.6),
                                   contrast_limit=(-0.1, 0.1), p=1.0),  # overexposure
        A.Compose([_fog(), _rain()]),  # fog+rain
        A.Compose([_fog(), _snow()]),  # fog+snow
        A.Compose([_rain(), _snow()]), # rain+snow
        A.MotionBlur(blur_limit=(15, 15), p=1.0),  # wind
    ]


class MultiWeather(ImageOnlyTransform):
    """随机或固定地对图像施加一种天气条件。

    weather_idx:
        None -> 每次随机选择一种天气（含 normal，用于训练）
        int  -> 固定使用该天气（0=normal, 1=fog, ...），用于测试
    """

    def __init__(self, weather_idx=None, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.weather_idx = weather_idx
        self.weather_ops = get_weather_ops()

    def apply(self, image, **params):
        if self.weather_idx is None:
            idx = random.randint(0, len(self.weather_ops) - 1)
        else:
            idx = self.weather_idx

        if idx == 0 or self.weather_ops[idx] is None:
            return image
        return self.weather_ops[idx](image=image)["image"]

    def get_transform_init_args_names(self):
        return ("weather_idx",)
