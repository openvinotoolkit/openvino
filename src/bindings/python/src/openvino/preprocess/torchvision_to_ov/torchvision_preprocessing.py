import numbers
import logging
import copy
import numpy as np
from typing import List, Dict
from abc import ABCMeta, abstractmethod
from typing import Callable, Any, Union, Tuple
from collections.abc import Sequence
from PIL import Image

import torch
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

import openvino.runtime as ov
import openvino.runtime.opset11 as ops
from openvino.runtime import Layout, Type
from openvino.runtime.utils.decorators import custom_preprocess_function
from openvino.preprocess import PrePostProcessor, ResizeAlgorithm, ColorFormat


TORCHTYPE_TO_OVTYPE = {
    float: ov.Type.f32,
    int: ov.Type.i32,
    bool: ov.Type.boolean,
    torch.float16: ov.Type.f16,
    torch.float32: ov.Type.f32,
    torch.float64: ov.Type.f64,
    torch.uint8: ov.Type.u8,
    torch.int8: ov.Type.i8,
    torch.int32: ov.Type.i32,
    torch.int64: ov.Type.i64,
    torch.bool: ov.Type.boolean,
    torch.DoubleTensor: ov.Type.f64,
    torch.FloatTensor: ov.Type.f32,
    torch.IntTensor: ov.Type.i32,
    torch.LongTensor: ov.Type.i64,
    torch.BoolTensor: ov.Type.boolean,
}


def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)
    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]
    if len(size) != 2:
        raise ValueError(error_msg)
    return size


def _NHWC_to_NCHW(input_shape):
    new_shape = copy.deepcopy(input_shape)
    new_shape[1] = input_shape[3]
    new_shape[2] = input_shape[1]
    new_shape[3] = input_shape[2]
    return new_shape


def _to_list(transform) -> List:
    if isinstance(transform, torch.nn.Sequential):
        return [t for t in transform]
    elif isinstance(transform, transforms.Compose):
        return transform.transforms
    else:
        raise TypeError(f"Unsupported transform type: {type(transform)}")


def _get_shape_layout_from_data(input_example: Union[torch.Tensor, np.ndarray, Image.Image]) -> Tuple[List, Layout]:
    """
    Disregards rank of shape and return
    """
    if isinstance(input_example, torch.Tensor):  # PyTorch
        shape = list(input_example.shape)
        layout = Layout("NCHW")
    elif isinstance(input_example, np.ndarray):  # OpenCV, numpy
        shape = list(input_example.shape)
        layout = Layout("NHWC")
    elif isinstance(input_example, Image.Image):  # PILLOW
        shape = list(np.array(input_example).shape)
        layout = Layout("NHWC")
    else:
        raise TypeError(f"Unsupported input type: {type(input_example)}")

    if len(shape) == 3:
        shape = [1] + shape

    return shape, layout


class TransformConverterBase(metaclass=ABCMeta):
    """Base class for an executor"""

    def __init__(self, **kwargs):
        """Constructor"""
        pass

    @abstractmethod
    def convert(self, input_idx: int, ppp: PrePostProcessor, transform):
        """Abstract method to run a command"""
        pass


class TransformConverterFactory:
    """The factory class for creating executors"""

    registry = {}
    """ Internal registry for available executors """

    @classmethod
    def register(cls, target_type=None) -> Callable:
        def inner_wrapper(wrapped_class: TransformConverterBase) -> Callable:
            registered_name = wrapped_class.__name__ if target_type is None else target_type.__name__
            if registered_name in cls.registry:
                logging.warning(f"Executor {registered_name} already exists. {wrapped_class.__name__} will replace it.")
            cls.registry[registered_name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def convert(cls, converter_type, *args, **kwargs):
        transform_name = converter_type.__name__
        if transform_name not in cls.registry:
            raise ValueError(f"{transform_name} is not supported.")

        converter = cls.registry[transform_name]()
        return converter.convert(*args, **kwargs)


@TransformConverterFactory.register(transforms.Normalize)
class _(TransformConverterBase):
    def convert(self, input_idx: int, ppp: PrePostProcessor, transform, meta: Dict):
        if transform.inplace:
            raise ValueError("Inplace Normaliziation is not supported.")
        ppp.input(input_idx).preprocess().mean(transform.mean).scale(transform.std)


@TransformConverterFactory.register(transforms.ConvertImageDtype)
class _(TransformConverterBase):
    def convert(self, input_idx: int, ppp: PrePostProcessor, transform, meta: Dict):
        ppp.input(input_idx).preprocess().convert_element_type(TORCHTYPE_TO_OVTYPE[transform.dtype])


@TransformConverterFactory.register(transforms.Grayscale)
class _(TransformConverterBase):
    def convert(self, input_idx: int, ppp: PrePostProcessor, transform, meta: Dict):
        input_shape = meta["input_shape"]
        layout = meta["layout"]

        input_shape[layout.get_index_by_name("C")] = 1

        ppp.input(input_idx).preprocess().convert_color(ColorFormat.GRAY)
        if transform.num_output_channels != 1:
            input_shape[layout.get_index_by_name("C")] = transform.num_output_channels

            @custom_preprocess_function
            def broadcast_node(output: ov.Output):
                return ops.broadcast(
                    data=output,
                    target_shape=input_shape,
                )
            ppp.input(input_idx).preprocess().custom(broadcast_node)

        meta["input_shape"] = input_shape
        

@TransformConverterFactory.register(transforms.Pad)
class _(TransformConverterBase):
    def convert(self, input_idx: int, ppp: PrePostProcessor, transform, meta: Dict):
        image_dimensions = list(meta["image_dimensions"])
        layout = meta["layout"]
        torch_padding = transform.padding
        pad_mode = transform.padding_mode

        if pad_mode == "constant":
            if isinstance(transform.fill, tuple):
                raise ValueError("Different fill values for R, G, B channels are not supported.")

        pads_begin = [0 for _ in meta["input_shape"]]
        pads_end = [0 for _ in meta["input_shape"]]

        # padding equal on all sides
        if isinstance(torch_padding, int):
            image_dimensions[0] += 2 * torch_padding
            image_dimensions[1] += 2 * torch_padding

            pads_begin[layout.get_index_by_name("H")] = torch_padding
            pads_begin[layout.get_index_by_name("W")] = torch_padding
            pads_end[layout.get_index_by_name("H")] = torch_padding
            pads_end[layout.get_index_by_name("W")] = torch_padding

        # padding different in horizontal and vertical axis
        elif len(torch_padding) == 2:
            image_dimensions[0] += sum(torch_padding)
            image_dimensions[1] += sum(torch_padding)

            pads_begin[layout.get_index_by_name("H")] = torch_padding[1]
            pads_begin[layout.get_index_by_name("W")] = torch_padding[0]
            pads_end[layout.get_index_by_name("H")] = torch_padding[1]
            pads_end[layout.get_index_by_name("W")] = torch_padding[0]

        # padding different on top, bottom, left and right of image
        else:
            image_dimensions[0] += torch_padding[1] + torch_padding[3]
            image_dimensions[1] += torch_padding[0] + torch_padding[2]

            pads_begin[layout.get_index_by_name("H")] = torch_padding[1]
            pads_begin[layout.get_index_by_name("W")] = torch_padding[0]
            pads_end[layout.get_index_by_name("H")] = torch_padding[3]
            pads_end[layout.get_index_by_name("W")] = torch_padding[2]

        @custom_preprocess_function
        def pad_node(output: ov.Output):
            return ops.pad(
                output,
                pad_mode=pad_mode,
                pads_begin=pads_begin,
                pads_end=pads_end,
                arg_pad_value=np.array(transform.fill, dtype=np.uint8) if pad_mode == "constant" else None,
            )

        ppp.input(input_idx).preprocess().custom(pad_node)
        meta["image_dimensions"] = tuple(image_dimensions)


@TransformConverterFactory.register(transforms.ToTensor)
class _(TransformConverterBase):
    def convert(self, input_idx: int, ppp: PrePostProcessor, transform, meta: Dict):
        input_shape = meta["input_shape"]
        layout = meta["layout"]

        ppp.input(input_idx).tensor().set_element_type(Type.u8).set_layout(Layout("NHWC")).set_color_format(ColorFormat.RGB)

        if layout == Layout("NHWC"):
            input_shape = _NHWC_to_NCHW(input_shape)
            layout = Layout("NCHW")
            ppp.input(input_idx).preprocess().convert_layout(layout)
        ppp.input(input_idx).preprocess().convert_element_type(Type.f32)
        ppp.input(input_idx).preprocess().scale(255.0)

        meta["input_shape"] = input_shape
        meta["layout"] = layout


@TransformConverterFactory.register(transforms.CenterCrop)
class _(TransformConverterBase):
    def convert(self, input_idx: int, ppp: PrePostProcessor, transform, meta: Dict):
        input_shape = meta["input_shape"]
        source_size = meta["image_dimensions"]
        target_size = _setup_size(transform.size, "Incorrect size type for CenterCrop operation")

        if target_size[0] > source_size[0] or target_size[1] > source_size[1]:
            ValueError(f"CenterCrop size={target_size} is greater than source_size={source_size}")

        bottom_left = []
        bottom_left.append(int((source_size[0] - target_size[0]) / 2))
        bottom_left.append(int((source_size[1] - target_size[1]) / 2))

        top_right = []
        top_right.append(min(bottom_left[0] + target_size[0], source_size[0] - 1))
        top_right.append(min(bottom_left[1] + target_size[1], source_size[1] - 1))

        bottom_left = [0] * len(input_shape[:-2]) + bottom_left if meta["layout"] == Layout("NCHW") else [0] + bottom_left + [0]
        top_right = input_shape[:-2] + top_right if meta["layout"] == Layout("NCHW") else input_shape[:1] + top_right + input_shape[-1:]

        ppp.input(input_idx).preprocess().crop(bottom_left, top_right)
        meta["image_dimensions"] = (target_size[-2], target_size[-1])


@TransformConverterFactory.register(transforms.Resize)
class _(TransformConverterBase):
    def convert(self, input_idx: int, ppp: PrePostProcessor, transform, meta: Dict):
        RESIZE_MODE_MAP = {
            InterpolationMode.BILINEAR: ResizeAlgorithm.RESIZE_BILINEAR_PILLOW,
            InterpolationMode.BICUBIC: ResizeAlgorithm.RESIZE_BICUBIC_PILLOW,
            InterpolationMode.NEAREST: ResizeAlgorithm.RESIZE_NEAREST
        }
        if transform.max_size:
            raise ValueError("Resize with max_size if not supported")

        h, w = _setup_size(transform.size, "Incorrect size type for Resize operation")

        ppp.input(input_idx).tensor().set_layout(Layout("NCHW"))

        input_shape = meta["input_shape"]

        input_shape[meta["layout"].get_index_by_name("H")] = -1
        input_shape[meta["layout"].get_index_by_name("W")] = -1

        ppp.input(input_idx).tensor().set_shape(input_shape)
        ppp.input(input_idx).preprocess().resize(RESIZE_MODE_MAP[transform.interpolation], h, w)
        meta["input_shape"] = input_shape
        meta["image_dimensions"] = (h, w)


def _from_torchvision(model: ov.Model, transform: Callable, input_example: Any, input_name: str = None) -> ov.Model:

    if input_name is not None:
        input_idx = next((i for i, p in enumerate(model.get_parameters()) if p.get_friendly_name() == input_name), None)
    else:
        if len(model.get_parameters()) == 1:
            input_idx = 0
        else:
            raise ValueError("Model contains multiple inputs. Please specify the name of the" "input to which prepocessing is added.")

    if input_idx is None:
        raise ValueError(f"Input with name {input_name} is not found")

    input_shape, layout = _get_shape_layout_from_data(input_example)

    ppp = PrePostProcessor(model)
    ppp.input(input_idx).tensor().set_layout(layout)
    ppp.input(input_idx).tensor().set_shape(input_shape)

    image_dimensions = [input_shape[layout.get_index_by_name("H")], input_shape[layout.get_index_by_name("W")]]
    global_meta = {
        "input_shape": input_shape,
        "image_dimensions": image_dimensions,
        "layout": layout,
    }

    for t in _to_list(transform):
        TransformConverterFactory.convert(type(t), input_idx, ppp, t, global_meta)

    updated_model = ppp.build()
    return updated_model
