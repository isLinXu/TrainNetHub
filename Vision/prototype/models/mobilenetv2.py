from functools import partial
from typing import Any, Optional

from torchvision.prototype.transforms import ImageNetEval
from torchvision.transforms.functional import InterpolationMode

from ...models.mobilenetv2 import MobileNetV2
from ._api import Weights, WeightEntry
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _deprecated_param, _deprecated_positional, _ovewrite_named_param


__all__ = ["MobileNetV2", "MobileNetV2Weights", "mobilenet_v2"]


class MobileNetV2Weights(Weights):
    ImageNet1K_RefV1 = WeightEntry(
        url="https://download.pytorch.org/models/mobilenet_v2-b0353104.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            "size": (224, 224),
            "categories": _IMAGENET_CATEGORIES,
            "interpolation": InterpolationMode.BILINEAR,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#mobilenetv2",
            "acc@1": 71.878,
            "acc@5": 90.286,
        },
        default=True,
    )


def mobilenet_v2(weights: Optional[MobileNetV2Weights] = None, progress: bool = True, **kwargs: Any) -> MobileNetV2:
    if type(weights) == bool and weights:
        _deprecated_positional(kwargs, "pretrained", "weights", True)
    if "pretrained" in kwargs:
        weights = _deprecated_param(kwargs, "pretrained", "weights", MobileNetV2Weights.ImageNet1K_RefV1)
    weights = MobileNetV2Weights.verify(weights)

    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = MobileNetV2(**kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model
