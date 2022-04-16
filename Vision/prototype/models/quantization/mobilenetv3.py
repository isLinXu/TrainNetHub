from functools import partial
from typing import Any, List, Optional, Union

import torch
from torchvision.prototype.transforms import ImageNetEval
from torchvision.transforms.functional import InterpolationMode

from ....models.quantization.mobilenetv3 import (
    InvertedResidualConfig,
    QuantizableInvertedResidual,
    QuantizableMobileNetV3,
    _replace_relu,
)
from .._api import Weights, WeightEntry
from .._meta import _IMAGENET_CATEGORIES
from .._utils import _deprecated_param, _deprecated_positional, _ovewrite_named_param
from ..mobilenetv3 import MobileNetV3LargeWeights, _mobilenet_v3_conf


__all__ = [
    "QuantizableMobileNetV3",
    "QuantizedMobileNetV3LargeWeights",
    "mobilenet_v3_large",
]


def _mobilenet_v3_model(
    inverted_residual_setting: List[InvertedResidualConfig],
    last_channel: int,
    weights: Optional[Weights],
    progress: bool,
    quantize: bool,
    **kwargs: Any,
) -> QuantizableMobileNetV3:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
        if "backend" in weights.meta:
            _ovewrite_named_param(kwargs, "backend", weights.meta["backend"])
    backend = kwargs.pop("backend", "qnnpack")

    model = QuantizableMobileNetV3(inverted_residual_setting, last_channel, block=QuantizableInvertedResidual, **kwargs)
    _replace_relu(model)

    if quantize:
        model.fuse_model()
        model.qconfig = torch.quantization.get_default_qat_qconfig(backend)
        torch.quantization.prepare_qat(model, inplace=True)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    if quantize:
        torch.quantization.convert(model, inplace=True)
        model.eval()

    return model


class QuantizedMobileNetV3LargeWeights(Weights):
    ImageNet1K_QNNPACK_RefV1 = WeightEntry(
        url="https://download.pytorch.org/models/quantized/mobilenet_v3_large_qnnpack-5bcacf28.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            "size": (224, 224),
            "categories": _IMAGENET_CATEGORIES,
            "interpolation": InterpolationMode.BILINEAR,
            "backend": "qnnpack",
            "quantization": "qat",
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#qat-mobilenetv3",
            "unquantized": MobileNetV3LargeWeights.ImageNet1K_RefV1,
            "acc@1": 73.004,
            "acc@5": 90.858,
        },
        default=True,
    )


def mobilenet_v3_large(
    weights: Optional[Union[QuantizedMobileNetV3LargeWeights, MobileNetV3LargeWeights]] = None,
    progress: bool = True,
    quantize: bool = False,
    **kwargs: Any,
) -> QuantizableMobileNetV3:
    if type(weights) == bool and weights:
        _deprecated_positional(kwargs, "pretrained", "weights", True)
    if "pretrained" in kwargs:
        default_value = (
            QuantizedMobileNetV3LargeWeights.ImageNet1K_QNNPACK_RefV1
            if quantize
            else MobileNetV3LargeWeights.ImageNet1K_RefV1
        )
        weights = _deprecated_param(kwargs, "pretrained", "weights", default_value)  # type: ignore[assignment]
    if quantize:
        weights = QuantizedMobileNetV3LargeWeights.verify(weights)
    else:
        weights = MobileNetV3LargeWeights.verify(weights)

    inverted_residual_setting, last_channel = _mobilenet_v3_conf("mobilenet_v3_large", **kwargs)
    return _mobilenet_v3_model(inverted_residual_setting, last_channel, weights, progress, quantize, **kwargs)
