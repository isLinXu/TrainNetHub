from functools import partial
from typing import Any, Optional

from torchvision.prototype.transforms import ImageNetEval
from torchvision.transforms.functional import InterpolationMode

from ...models.shufflenetv2 import ShuffleNetV2
from ._api import Weights, WeightEntry
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _deprecated_param, _deprecated_positional, _ovewrite_named_param


__all__ = [
    "ShuffleNetV2",
    "ShuffleNetV2_x0_5Weights",
    "ShuffleNetV2_x1_0Weights",
    "ShuffleNetV2_x1_5Weights",
    "ShuffleNetV2_x2_0Weights",
    "shufflenet_v2_x0_5",
    "shufflenet_v2_x1_0",
    "shufflenet_v2_x1_5",
    "shufflenet_v2_x2_0",
]


def _shufflenetv2(
    weights: Optional[Weights],
    progress: bool,
    *args: Any,
    **kwargs: Any,
) -> ShuffleNetV2:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ShuffleNetV2(*args, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


_COMMON_META = {
    "size": (224, 224),
    "categories": _IMAGENET_CATEGORIES,
    "interpolation": InterpolationMode.BILINEAR,
    "recipe": "https://github.com/barrh/Shufflenet-v2-Pytorch/tree/v0.1.0",
}


class ShuffleNetV2_x0_5Weights(Weights):
    ImageNet1K_Community = WeightEntry(
        url="https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "acc@1": 69.362,
            "acc@5": 88.316,
        },
        default=True,
    )


class ShuffleNetV2_x1_0Weights(Weights):
    ImageNet1K_Community = WeightEntry(
        url="https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "acc@1": 60.552,
            "acc@5": 81.746,
        },
        default=True,
    )


class ShuffleNetV2_x1_5Weights(Weights):
    pass


class ShuffleNetV2_x2_0Weights(Weights):
    pass


def shufflenet_v2_x0_5(
    weights: Optional[ShuffleNetV2_x0_5Weights] = None, progress: bool = True, **kwargs: Any
) -> ShuffleNetV2:
    if type(weights) == bool and weights:
        _deprecated_positional(kwargs, "pretrained", "weights", True)
    if "pretrained" in kwargs:
        weights = _deprecated_param(kwargs, "pretrained", "weights", ShuffleNetV2_x0_5Weights.ImageNet1K_Community)
    weights = ShuffleNetV2_x0_5Weights.verify(weights)

    return _shufflenetv2(weights, progress, [4, 8, 4], [24, 48, 96, 192, 1024], **kwargs)


def shufflenet_v2_x1_0(
    weights: Optional[ShuffleNetV2_x1_0Weights] = None, progress: bool = True, **kwargs: Any
) -> ShuffleNetV2:
    if type(weights) == bool and weights:
        _deprecated_positional(kwargs, "pretrained", "weights", True)
    if "pretrained" in kwargs:
        weights = _deprecated_param(kwargs, "pretrained", "weights", ShuffleNetV2_x1_0Weights.ImageNet1K_Community)
    weights = ShuffleNetV2_x1_0Weights.verify(weights)

    return _shufflenetv2(weights, progress, [4, 8, 4], [24, 116, 232, 464, 1024], **kwargs)


def shufflenet_v2_x1_5(
    weights: Optional[ShuffleNetV2_x1_5Weights] = None, progress: bool = True, **kwargs: Any
) -> ShuffleNetV2:
    if type(weights) == bool and weights:
        _deprecated_positional(kwargs, "pretrained", "weights", True)
    if "pretrained" in kwargs:
        weights = _deprecated_param(kwargs, "pretrained", "weights", None)
    weights = ShuffleNetV2_x1_5Weights.verify(weights)

    return _shufflenetv2(weights, progress, [4, 8, 4], [24, 176, 352, 704, 1024], **kwargs)


def shufflenet_v2_x2_0(
    weights: Optional[ShuffleNetV2_x2_0Weights] = None, progress: bool = True, **kwargs: Any
) -> ShuffleNetV2:
    if type(weights) == bool and weights:
        _deprecated_positional(kwargs, "pretrained", "weights", True)
    if "pretrained" in kwargs:
        weights = _deprecated_param(kwargs, "pretrained", "weights", None)
    weights = ShuffleNetV2_x2_0Weights.verify(weights)

    return _shufflenetv2(weights, progress, [4, 8, 4], [24, 244, 488, 976, 2048], **kwargs)
