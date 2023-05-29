from typing import List
from abc import ABCMeta, abstractmethod
from typing import Callable, Any
from enum import Enum
import numbers
from collections.abc import Sequence
import logging
import copy
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

import openvino.runtime as ov
from openvino.preprocess import PrePostProcessor, ResizeAlgorithm, ColorFormat
from openvino.runtime import Core, Layout, Type
import openvino.runtime.opset11 as ops
from openvino.runtime.utils.decorators import custom_preprocess_function

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


def _change_layout_shape(original_shape):
    new_shape = copy.deepcopy(original_shape)
    new_shape[1] = original_shape[3]
    new_shape[2] = original_shape[1]
    new_shape[3] = original_shape[2]
    return new_shape


class TransformConverterBase(metaclass=ABCMeta):
    """ Base class for an executor """
 
    def __init__(self, **kwargs):
        """ Constructor """
        pass
 
    @abstractmethod
    def convert(self, input_idx: int, ppp: PrePostProcessor, transform):
        """ Abstract method to run a command """
        return False
 

class TransformConverterFactory:
    """ The factory class for creating executors"""
 
    registry = {}
    """ Internal registry for available executors """
 
    @classmethod
    def register(cls, target_type=None) -> Callable:
        def inner_wrapper(wrapped_class: TransformConverterBase) -> Callable:
            registered_name = wrapped_class.__name__ if target_type == None else target_type.__name__
            if registered_name in cls.registry:
                logging.warning('Executor %s already exists. Will replace it', registered_name)
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


### Converters definition 
@TransformConverterFactory.register(transforms.Normalize)
class NormalizeConverter(TransformConverterBase):
    def convert(self, input_idx: int, ppp: PrePostProcessor, transform, meta=None):
        mean = transform.mean
        scale = transform.std  # [1/std for std in transform.std]
        ppp.input(input_idx).preprocess().mean(mean).scale(scale)
        return None


@TransformConverterFactory.register(transforms.ConvertImageDtype)
class NormalizeConverter(TransformConverterBase):
    def convert(self, input_idx: int, ppp: PrePostProcessor, transform, meta=None):
        ppp.input(input_idx).preprocess().convert_element_type(TORCHTYPE_TO_OVTYPE[transform.dtype])
        return None


@TransformConverterFactory.register(transforms.Grayscale)
class NormalizeConverter(TransformConverterBase):
    def convert(self, input_idx: int, ppp: PrePostProcessor, transform, meta=None):
        if transform.num_output_chanels != 1:
            raise ValueError('OpenVINO does not support multi-channel grayscale output')  # TODO: Tomek

        ppp.input(input_idx).preprocess().convert_color(ColorFormat.GRAY)
        return None


@TransformConverterFactory.register(transforms.Pad)
class NormalizeConverter(TransformConverterBase):
    # https://pytorch.org/vision/stable/generated/torchvision.transforms.Pad.html#torchvision.transforms.Pad
    def convert(self, input_idx: int, ppp: PrePostProcessor, transform, meta=None):
        original_shape = meta["original_shape"]
        current_shape = list(meta["current_shape"])
        layout = meta["layout"]
        torch_padding = transform.padding
        pad_mode = transform.padding_mode

        if pad_mode == "constant":
            if isinstance(transform.fill, tuple):  # TODO: Tomek
                raise ValueError("Different fill values for R, G, B channels are not supported.")

        pads_begin = [0 for _ in original_shape]
        pads_end = [0 for _ in original_shape]

        # padding equal on all sides
        if isinstance(torch_padding, int):
            current_shape[0] += 2 * torch_padding
            current_shape[1] += 2 * torch_padding

            pads_begin[layout.get_index_by_name("H")] = torch_padding
            pads_begin[layout.get_index_by_name("W")] = torch_padding
            pads_end[layout.get_index_by_name("H")] = torch_padding
            pads_end[layout.get_index_by_name("W")] = torch_padding

        # padding different in horizontal and vertical axis
        elif len(torch_padding) == 2:
            current_shape[0] += sum(torch_padding)
            current_shape[1] += sum(torch_padding)

            pads_begin[layout.get_index_by_name("H")] = torch_padding[1]
            pads_begin[layout.get_index_by_name("W")] = torch_padding[0]
            pads_end[layout.get_index_by_name("H")] = torch_padding[1]
            pads_end[layout.get_index_by_name("W")] = torch_padding[0]

        # padding different on top, bottom, left and right of image
        else:
            current_shape[0] += torch_padding[1] + torch_padding[3]
            current_shape[1] += torch_padding[0] + torch_padding[2]

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
                arg_pad_value=np.array(transform.fill, dtype=np.uint8) if pad_mode == "constant" else None
                )

        ppp.input(input_idx).preprocess().custom(pad_node)
        meta["current_shape"] = tuple(current_shape)
        return meta


@TransformConverterFactory.register(transforms.ToTensor)
class NormalizeConverter(TransformConverterBase):
    def convert(self, input_idx: int, ppp: PrePostProcessor, transform, meta=None):
        original_shape = meta["original_shape"]
        layout = meta["layout"]

        ppp.input(input_idx).tensor() \
        .set_element_type(Type.u8) \
        .set_layout(Layout('NHWC')) \
        .set_color_format(ColorFormat.RGB)

        if layout == Layout("NHWC"):
            original_shape = _change_layout_shape(original_shape)
            layout = Layout("NCHW")
            ppp.input(input_idx).preprocess().convert_layout(layout)
        ppp.input(input_idx).preprocess().convert_element_type(Type.f32)
        ppp.input(input_idx).preprocess().scale(255.0)

        meta["original_shape"] = original_shape
        meta["layout"] = layout
        return meta


@TransformConverterFactory.register(transforms.CenterCrop)
class NormalizeConverter(TransformConverterBase):
    def convert(self, input_idx: int, ppp: PrePostProcessor, transform, meta=None):
        original_shape = meta["original_shape"]
        layout = meta["layout"]
        
        source_size = meta["current_shape"]
        target_size = _setup_size(transform.size, "Incorrect size type for CenterCrop operation")        

        if target_size[0] > source_size[0] or target_size[1] > source_size[1]:
            ValueError(f"CenterCrop size={target_size} is greater than source_size={source_size}")
        
        bottom_left = []
        bottom_left.append(int((source_size[0] - target_size[0]) / 2))
        bottom_left.append(int((source_size[1] - target_size[1]) / 2))
        
        top_right = []
        top_right.append(min(bottom_left[0] + target_size[0], source_size[0] - 1))
        top_right.append(min(bottom_left[1] + target_size[1], source_size[1] - 1))

        bottom_left = [0]*len(original_shape[:-2]) + bottom_left if layout == Layout("NCHW") else [0] + bottom_left + [0]
        top_right = original_shape[:-2] + top_right if layout == Layout("NCHW") else original_shape[:1] + top_right + original_shape[-1:]

        ppp.input(input_idx).preprocess().crop(bottom_left, top_right)
        meta["current_shape"] = (target_size[-2],target_size[-1])
        return meta


@TransformConverterFactory.register(transforms.Resize)
class NormalizeConverter(TransformConverterBase):
    RESIZE_MODE_MAP = {
        InterpolationMode.BILINEAR: ResizeAlgorithm.RESIZE_LINEAR,
        InterpolationMode.BICUBIC: ResizeAlgorithm.RESIZE_CUBIC,
        InterpolationMode.NEAREST: ResizeAlgorithm.RESIZE_NEAREST,
    }
    def convert(self, input_idx: int, ppp: PrePostProcessor, transform, meta=None):
        if transform.max_size != None:
            raise ValueError('Resize with max_size if not supported')  # TODO: Tomek

        mode = transform.interpolation
        size = transform.size
        h, w = _setup_size(size, "Incorrect size type for Resize operation")

        ppp.input(input_idx).tensor().set_layout(Layout('NCHW'))

        layout = meta["layout"] 
        original_shape = meta["original_shape"]

        if layout == Layout("NHWC"):
            original_shape[1] = -1
            original_shape[2] = -1
        else:
            original_shape[2] = -1
            original_shape[3] = -1

        ppp.input(input_idx).tensor().set_shape(original_shape)
        ppp.input(input_idx).preprocess().resize(NormalizeConverter.RESIZE_MODE_MAP[mode], h, w)
        meta["original_shape"] = original_shape
        meta["current_shape"] = (h, w)

        return meta

def _to_list(transform) -> List:
        if isinstance(transform, torch.nn.Sequential):
            return [t for t in transform]
        elif isinstance(transform, transforms.Compose):
            return transform.transforms
        else:
            raise TypeError(f"Unsupported transform type: {type(transform)}")

def _get_shape_layout_from_data(input_example):
    """
    Disregards rank of shape and return 
    """
    shape = None
    layout = None
    if isinstance(input_example, torch.Tensor): # PyTorch
        shape = list(input_example.shape)
        layout = Layout("NCHW")
    elif isinstance(input_example, np.ndarray): # OpenCV, numpy
        shape = list(input_example.shape)
        layout = Layout("NHWC")
    elif isinstance(input_example, Image.Image): # PILLOW
        shape = list(np.array(input_example).shape)
        layout = Layout("NHWC")
    else:
        raise TypeError(f"Unsupported input type: {type(input_example)}")

    if len(shape) == 3:
        shape = [1] + shape

    return shape, layout

def from_torchvision(model: ov.Model, transform: Callable, input_example: Any, 
                    input_name: str = None) -> ov.Model:
        transform_list = _to_list(transform)

        if input_name is not None:
            input_idx = next((i for i, p in enumerate(model.get_parameters()) if p.get_friendly_name() == input_name), None)
        else:
            if len(model.get_parameters()) == 1:
                input_idx = 0
            else:
                raise ValueError("Model contains multiple inputs. Please specify the name of the"
                                "input to which prepocessing is added.")

        if input_idx is None:
            raise ValueError(f"Input with name {input_name} is not found")

        original_shape, layout = _get_shape_layout_from_data(input_example)

        ppp = PrePostProcessor(model)
        ppp.input(input_idx).tensor().set_layout(layout) 
        ppp.input(input_idx).tensor().set_shape(original_shape)

        if layout == Layout("NHWC"):
            current_shape = [original_shape[1], original_shape[2]]
        else:
            current_shape = [original_shape[2], original_shape[3]]
        global_meta = {
            "original_shape": original_shape,
            "current_shape": current_shape,
            "layout": layout,
            "has_totensor": any(isinstance(item, transforms.ToTensor) for item in transform_list)
        }

        for t in transform_list:
            _ = TransformConverterFactory.convert(type(t), input_idx, ppp, t, global_meta)

        updated_model = ppp.build()
        return updated_model
