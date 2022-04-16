import warnings
from functools import partial
from typing import Any, Optional, Union

from torchvision.prototype.transforms import ImageNetEval
from torchvision.transforms.functional import InterpolationMode

from ....models.quantization.googlenet import (
    QuantizableGoogLeNet,
    _replace_relu,
    quantize_model,
)
from .._api import Weights, WeightEntry
from .._meta import _IMAGENET_CATEGORIES
from .._utils import _deprecated_param, _deprecated_positional, _ovewrite_named_param
from ..googlenet import GoogLeNetWeights


__all__ = [
    "QuantizableGoogLeNet",
    "QuantizedGoogLeNetWeights",
    "googlenet",
]


class QuantizedGoogLeNetWeights(Weights):
    ImageNet1K_FBGEMM_TFV1 = WeightEntry(
        url="https://download.pytorch.org/models/quantized/googlenet_fbgemm-c00238cf.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            "size": (224, 224),
            "categories": _IMAGENET_CATEGORIES,
            "interpolation": InterpolationMode.BILINEAR,
            "backend": "fbgemm",
            "quantization": "ptq",
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#post-training-quantized-models",
            "unquantized": GoogLeNetWeights.ImageNet1K_TFV1,
            "acc@1": 69.826,
            "acc@5": 89.404,
        },
        default=True,
    )


def googlenet(
    weights: Optional[Union[QuantizedGoogLeNetWeights, GoogLeNetWeights]] = None,
    progress: bool = True,
    quantize: bool = False,
    **kwargs: Any,
) -> QuantizableGoogLeNet:
    if type(weights) == bool and weights:
        _deprecated_positional(kwargs, "pretrained", "weights", True)
    if "pretrained" in kwargs:
        default_value = (
            QuantizedGoogLeNetWeights.ImageNet1K_FBGEMM_TFV1 if quantize else GoogLeNetWeights.ImageNet1K_TFV1
        )
        weights = _deprecated_param(kwargs, "pretrained", "weights", default_value)  # type: ignore[assignment]
    if quantize:
        weights = QuantizedGoogLeNetWeights.verify(weights)
    else:
        weights = GoogLeNetWeights.verify(weights)

    original_aux_logits = kwargs.get("aux_logits", False)
    if weights is not None:
        if "transform_input" not in kwargs:
            _ovewrite_named_param(kwargs, "transform_input", True)
        _ovewrite_named_param(kwargs, "aux_logits", True)
        _ovewrite_named_param(kwargs, "init_weights", False)
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
        if "backend" in weights.meta:
            _ovewrite_named_param(kwargs, "backend", weights.meta["backend"])
    backend = kwargs.pop("backend", "fbgemm")

    model = QuantizableGoogLeNet(**kwargs)
    _replace_relu(model)
    if quantize:
        quantize_model(model, backend)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))
        if not original_aux_logits:
            model.aux_logits = False
            model.aux1 = None  # type: ignore[assignment]
            model.aux2 = None  # type: ignore[assignment]
        else:
            warnings.warn(
                "auxiliary heads in the pretrained googlenet model are NOT pretrained, so make sure to train them"
            )

    return model
