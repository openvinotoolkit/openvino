# type: ignore
from __future__ import annotations
import collections.abc
import numpy
import openvino._pyopenvino
import typing
"""
Package openvino.preprocess that wraps ov::preprocess
"""
__all__ = ['BGR', 'BGRX', 'CONSTANT', 'ColorFormat', 'GRAY', 'I420_SINGLE_PLANE', 'I420_THREE_PLANES', 'InputInfo', 'InputModelInfo', 'InputTensorInfo', 'NV12_SINGLE_PLANE', 'NV12_TWO_PLANES', 'OutputInfo', 'OutputModelInfo', 'OutputTensorInfo', 'PaddingMode', 'PostProcessSteps', 'PrePostProcessor', 'PreProcessSteps', 'REFLECT', 'RESIZE_BICUBIC_PILLOW', 'RESIZE_BILINEAR_PILLOW', 'RESIZE_CUBIC', 'RESIZE_LINEAR', 'RESIZE_NEAREST', 'RGB', 'RGBX', 'ResizeAlgorithm', 'SYMMETRIC', 'UNDEFINED']
class ColorFormat:
    """
    Members:
    
      UNDEFINED
    
      NV12_SINGLE_PLANE
    
      NV12_TWO_PLANES
    
      I420_SINGLE_PLANE
    
      I420_THREE_PLANES
    
      RGB
    
      BGR
    
      GRAY
    
      RGBX
    
      BGRX
    """
    BGR: typing.ClassVar[ColorFormat]  # value = <ColorFormat.BGR: 6>
    BGRX: typing.ClassVar[ColorFormat]  # value = <ColorFormat.BGRX: 9>
    GRAY: typing.ClassVar[ColorFormat]  # value = <ColorFormat.GRAY: 7>
    I420_SINGLE_PLANE: typing.ClassVar[ColorFormat]  # value = <ColorFormat.I420_SINGLE_PLANE: 3>
    I420_THREE_PLANES: typing.ClassVar[ColorFormat]  # value = <ColorFormat.I420_THREE_PLANES: 4>
    NV12_SINGLE_PLANE: typing.ClassVar[ColorFormat]  # value = <ColorFormat.NV12_SINGLE_PLANE: 1>
    NV12_TWO_PLANES: typing.ClassVar[ColorFormat]  # value = <ColorFormat.NV12_TWO_PLANES: 2>
    RGB: typing.ClassVar[ColorFormat]  # value = <ColorFormat.RGB: 5>
    RGBX: typing.ClassVar[ColorFormat]  # value = <ColorFormat.RGBX: 8>
    UNDEFINED: typing.ClassVar[ColorFormat]  # value = <ColorFormat.UNDEFINED: 0>
    __members__: typing.ClassVar[dict[str, ColorFormat]]  # value = {'UNDEFINED': <ColorFormat.UNDEFINED: 0>, 'NV12_SINGLE_PLANE': <ColorFormat.NV12_SINGLE_PLANE: 1>, 'NV12_TWO_PLANES': <ColorFormat.NV12_TWO_PLANES: 2>, 'I420_SINGLE_PLANE': <ColorFormat.I420_SINGLE_PLANE: 3>, 'I420_THREE_PLANES': <ColorFormat.I420_THREE_PLANES: 4>, 'RGB': <ColorFormat.RGB: 5>, 'BGR': <ColorFormat.BGR: 6>, 'GRAY': <ColorFormat.GRAY: 7>, 'RGBX': <ColorFormat.RGBX: 8>, 'BGRX': <ColorFormat.BGRX: 9>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class InputInfo:
    """
    openvino.preprocess.InputInfo wraps ov::preprocess::InputInfo
    """
    def model(self) -> InputModelInfo:
        ...
    def preprocess(self) -> PreProcessSteps:
        ...
    def tensor(self) -> InputTensorInfo:
        ...
class InputModelInfo:
    """
    openvino.preprocess.InputModelInfo wraps ov::preprocess::InputModelInfo
    """
    def set_layout(self, layout: openvino._pyopenvino.Layout) -> InputModelInfo:
        """
                    Set layout for input model
                    :param layout: layout to be set
                    :type layout: Union[str, openvino.Layout]
        """
class InputTensorInfo:
    """
    openvino.preprocess.InputTensorInfo wraps ov::preprocess::InputTensorInfo
    """
    def set_color_format(self, format: ColorFormat, sub_names: collections.abc.Sequence[str] = []) -> InputTensorInfo:
        ...
    def set_element_type(self, type: openvino._pyopenvino.Type) -> InputTensorInfo:
        """
                    Set initial client's tensor element type. If type is not the same as model's element type,
                    conversion of element type will be done automatically.
        
                    :param type: Client's input tensor element type.
                    :type type: openvino.Type
                    :return: Reference to itself, allows chaining of calls in client's code in a builder-like manner.
                    :rtype: openvino.preprocess.InputTensorInfo
        """
    @typing.overload
    def set_from(self, tensor: openvino._pyopenvino.Tensor) -> InputTensorInfo:
        """
                    Helper function to reuse element type and shape from user's created tensor. Overwrites previously
                    set shape and element type via `set_shape` and `set_element_type' methods. This method should be
                    used only in case if tensor is already known and available before.
        
                    :param tensor: User's created tensor
                    :type type: openvino.Tensor
                    :return: Reference to itself, allows chaining of calls in client's code in a builder-like manner.
                    :rtype: openvino.preprocess.InputTensorInfo
        """
    @typing.overload
    def set_from(self, tensor: numpy.ndarray[typing.Any, numpy.dtype[typing.Any]]) -> InputTensorInfo:
        """
                    Helper function to reuse element type and shape from user's created tensor. Overwrites previously
                    set shape and element type via `set_shape` and `set_element_type' methods. This method should be
                    used only in case if tensor is already known and available before.
        
                    :param tensor: User's created numpy array
                    :type type: numpy.ndarray
                    :return: Reference to itself, allows chaining of calls in client's code in a builder-like manner.
                    :rtype: openvino.preprocess.InputTensorInfo
        """
    def set_layout(self, layout: openvino._pyopenvino.Layout) -> InputTensorInfo:
        """
                    Set layout for input tensor info
        
                    :param layout: layout to be set
                    :type layout: Union[str, openvino.Layout]
        """
    def set_memory_type(self, memory_type: str) -> InputTensorInfo:
        ...
    @typing.overload
    def set_shape(self, shape: openvino._pyopenvino.PartialShape) -> InputTensorInfo:
        ...
    @typing.overload
    def set_shape(self, shape: collections.abc.Sequence[typing.SupportsInt]) -> InputTensorInfo:
        ...
    def set_spatial_dynamic_shape(self) -> InputTensorInfo:
        ...
    def set_spatial_static_shape(self, height: typing.SupportsInt, width: typing.SupportsInt) -> InputTensorInfo:
        ...
class OutputInfo:
    """
    openvino.preprocess.OutputInfo wraps ov::preprocess::OutputInfo
    """
    def model(self) -> OutputModelInfo:
        ...
    def postprocess(self) -> PostProcessSteps:
        ...
    def tensor(self) -> OutputTensorInfo:
        ...
class OutputModelInfo:
    """
    openvino.preprocess.OutputModelInfo wraps ov::preprocess::OutputModelInfo
    """
    def set_layout(self, layout: openvino._pyopenvino.Layout) -> OutputModelInfo:
        """
                    Set layout for output model info
        
                    :param layout: layout to be set
                    :type layout: Union[str, openvino.Layout]
        """
class OutputTensorInfo:
    """
    openvino.preprocess.OutputTensorInfo wraps ov::preprocess::OutputTensorInfo
    """
    def set_element_type(self, type: openvino._pyopenvino.Type) -> OutputTensorInfo:
        """
                    Set client's output tensor element type. If type is not the same as model's element type,
                    conversion of element type will be done automatically.
        
                    :param type: Client's output tensor element type.
                    :type type: openvino.Type
                    :return: Reference to itself to allow chaining of calls in client's code in a builder-like manner.
                    :rtype: openvino.preprocess.OutputTensorInfo
        """
    def set_layout(self, layout: openvino._pyopenvino.Layout) -> OutputTensorInfo:
        """
                    Set layout for output tensor info
        
                    :param layout: layout to be set
                    :type layout: Union[str, openvino.Layout]
        """
class PaddingMode:
    """
    Members:
    
      CONSTANT
    
      REFLECT
    
      SYMMETRIC
    """
    CONSTANT: typing.ClassVar[PaddingMode]  # value = <PaddingMode.CONSTANT: 0>
    REFLECT: typing.ClassVar[PaddingMode]  # value = <PaddingMode.REFLECT: 2>
    SYMMETRIC: typing.ClassVar[PaddingMode]  # value = <PaddingMode.SYMMETRIC: 3>
    __members__: typing.ClassVar[dict[str, PaddingMode]]  # value = {'CONSTANT': <PaddingMode.CONSTANT: 0>, 'REFLECT': <PaddingMode.REFLECT: 2>, 'SYMMETRIC': <PaddingMode.SYMMETRIC: 3>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class PostProcessSteps:
    """
    openvino.preprocess.PostprocessSteps wraps ov::preprocess::PostProcessSteps
    """
    def convert_element_type(self, type: openvino._pyopenvino.Type = ...) -> PostProcessSteps:
        """
                    Converts tensor element type to specified type.
                    Tensor must have openvino.Type data type.
        
                    :param type: Destination type. If not specified, type will be taken from model output's element type.
                    :type type: openvino.Type
                    :return: Reference to itself to allow chaining of calls in client's code in a builder-like manner.
                    :rtype: openvino.preprocess.PostProcessSteps
        """
    @typing.overload
    def convert_layout(self, dst_layout: openvino._pyopenvino.Layout) -> PostProcessSteps:
        ...
    @typing.overload
    def convert_layout(self, dims: collections.abc.Sequence[typing.SupportsInt]) -> PostProcessSteps:
        ...
    def custom(self, operation: collections.abc.Callable) -> PostProcessSteps:
        """
                    Adds custom postprocessing operation.
        
                    :param operation: Python's function which takes `openvino.Output` as input argument and returns`openvino.Output`.
                    :type operation: function
                    :return: Reference to itself, allows chaining of calls in client's code in a builder-like manner.
                    :rtype: openvino.preprocess.PreProcessSteps
        """
class PrePostProcessor:
    """
    openvino.preprocess.PrePostProcessor wraps ov::preprocess::PrePostProcessor
    """
    def __init__(self, model: typing.Any) -> None:
        """
                     It creates PrePostProcessor.
        """
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    def build(self) -> typing.Any:
        ...
    @typing.overload
    def input(self) -> InputInfo:
        ...
    @typing.overload
    def input(self, tensor_name: str) -> InputInfo:
        ...
    @typing.overload
    def input(self, input_index: typing.SupportsInt) -> InputInfo:
        ...
    @typing.overload
    def output(self) -> OutputInfo:
        ...
    @typing.overload
    def output(self, tensor_name: str) -> OutputInfo:
        ...
    @typing.overload
    def output(self, output_index: typing.SupportsInt) -> OutputInfo:
        ...
class PreProcessSteps:
    """
    openvino.preprocess.PreProcessSteps wraps ov::preprocess::PreProcessSteps
    """
    def convert_color(self, dst_format: ColorFormat) -> PreProcessSteps:
        ...
    def convert_element_type(self, type: openvino._pyopenvino.Type = ...) -> PreProcessSteps:
        """
                    Converts input tensor element type to specified type.
                    Input tensor must have openvino.Type data type.
        
                    :param type: Destination type. If not specified, type will be taken from model input's element type
                    :type type: openvino.Type
                    :return: Reference to itself to allow chaining of calls in client's code in a builder-like manner.
                    :rtype: openvino.preprocess.PreProcessSteps
        """
    @typing.overload
    def convert_layout(self, dst_layout: openvino._pyopenvino.Layout) -> PreProcessSteps:
        ...
    @typing.overload
    def convert_layout(self, dims: collections.abc.Sequence[typing.SupportsInt]) -> PreProcessSteps:
        ...
    def crop(self, begin: collections.abc.Sequence[typing.SupportsInt], end: collections.abc.Sequence[typing.SupportsInt]) -> PreProcessSteps:
        ...
    def custom(self, operation: collections.abc.Callable) -> PreProcessSteps:
        """
                    Adds custom preprocessing operation.
        
                    :param operation: Python's function which takes `openvino.Output` as input argument and returns`openvino.Output`.
                    :type operation: function
                    :return: Reference to itself, allows chaining of calls in client's code in a builder-like manner.
                    :rtype: openvino.preprocess.PreProcessSteps
        """
    @typing.overload
    def mean(self, value: typing.SupportsFloat) -> PreProcessSteps:
        """
                    Subtracts single float value from each element in input tensor.
                    Input tensor must have ov.Type.f32 data type.
        
                    :param value: Value to subtract.
                    :type value: float
                    :return: Reference to itself to allow chaining of calls in client's code in a builder-like manner.
                    :rtype: openvino.preprocess.PreProcessSteps
        """
    @typing.overload
    def mean(self, values: collections.abc.Sequence[typing.SupportsFloat]) -> PreProcessSteps:
        """
                    Subtracts a given single float value from each element in a given channel from input tensor.
                    Input tensor must have ov.Type.f32 data type.
        
                    :param values: Values to subtract.
                    :type values: list[float]
                    :return: Reference to itself to allow chaining of calls in client's code in a builder-like manner.
                    :rtype: openvino.preprocess.PreProcessSteps
        """
    @typing.overload
    def pad(self, pads_begin: collections.abc.Sequence[typing.SupportsInt], pads_end: collections.abc.Sequence[typing.SupportsInt], value: typing.SupportsFloat, mode: PaddingMode) -> PreProcessSteps:
        """
                    Adds padding preprocessing operation.
        
                    :param pads_begin: Number of elements matches the number of indices in data attribute. Specifies the number of padding elements at the ending of each axis.
                    :type pads_begin: 1D tensor of type T_INT.
                    :param pads_end: Number of elements matches the number of indices in data attribute. Specifies the number of padding elements at the ending of each axis.
                    :type pads_end: 1D tensor of type T_INT.
                    :param value: All new elements are populated with this value or with 0 if input not provided. Shouldn't be set for other pad_mode values.
                    :type value: scalar tensor of type T.
                    :param mode: pad_mode specifies the method used to generate new element values.
                    :type mode: string
                    :return: Reference to itself, allows chaining of calls in client's code in a builder-like manner.
                    :rtype: openvino.preprocess.PreProcessSteps
        """
    @typing.overload
    def pad(self, pads_begin: collections.abc.Sequence[typing.SupportsInt], pads_end: collections.abc.Sequence[typing.SupportsInt], value: collections.abc.Sequence[typing.SupportsFloat], mode: PaddingMode) -> PreProcessSteps:
        """
                    Adds padding preprocessing operation.
        
                    :param pads_begin: Number of elements matches the number of indices in data attribute. Specifies the number of padding elements at the ending of each axis.
                    :type pads_begin: 1D tensor of type T_INT.
                    :param pads_end: Number of elements matches the number of indices in data attribute. Specifies the number of padding elements at the ending of each axis.
                    :type pads_end: 1D tensor of type T_INT.
                    :param value: All new elements are populated with this value or with 0 if input not provided. Shouldn't be set for other pad_mode values.
                    :type value: scalar tensor of type T.
                    :param mode: pad_mode specifies the method used to generate new element values.
                    :type mode: string
                    :return: Reference to itself, allows chaining of calls in client's code in a builder-like manner.
                    :rtype: openvino.PreProcessSteps
        """
    @typing.overload
    def resize(self, alg: ResizeAlgorithm, dst_height: typing.SupportsInt, dst_width: typing.SupportsInt) -> PreProcessSteps:
        ...
    @typing.overload
    def resize(self, alg: ResizeAlgorithm) -> PreProcessSteps:
        ...
    def reverse_channels(self) -> PreProcessSteps:
        ...
    @typing.overload
    def scale(self, value: typing.SupportsFloat) -> PreProcessSteps:
        """
                    Divides each element in input tensor by specified constant float value.
                    Input tensor must have ov.Type.f32 data type.
        
                    :param value: Value used in division.
                    :type value: float
                    :return: Reference to itself to allow chaining of calls in client's code in a builder-like manner.
                    :rtype: openvino.preprocess.PreProcessSteps
        """
    @typing.overload
    def scale(self, values: collections.abc.Sequence[typing.SupportsFloat]) -> PreProcessSteps:
        """
                    Divides each element in a given channel from input tensor by a given single float value.
                    Input tensor must have ov.Type.f32 data type.
        
                    :param values: Values which are used in division.
                    :type values: list[float]
                    :return: Reference to itself to allow chaining of calls in client's code in a builder-like manner.
                    :rtype: openvino.preprocess.PreProcessSteps
        """
class ResizeAlgorithm:
    """
    Members:
    
      RESIZE_LINEAR
    
      RESIZE_CUBIC
    
      RESIZE_NEAREST
    
      RESIZE_BILINEAR_PILLOW
    
      RESIZE_BICUBIC_PILLOW
    """
    RESIZE_BICUBIC_PILLOW: typing.ClassVar[ResizeAlgorithm]  # value = <ResizeAlgorithm.RESIZE_BICUBIC_PILLOW: 4>
    RESIZE_BILINEAR_PILLOW: typing.ClassVar[ResizeAlgorithm]  # value = <ResizeAlgorithm.RESIZE_BILINEAR_PILLOW: 3>
    RESIZE_CUBIC: typing.ClassVar[ResizeAlgorithm]  # value = <ResizeAlgorithm.RESIZE_CUBIC: 1>
    RESIZE_LINEAR: typing.ClassVar[ResizeAlgorithm]  # value = <ResizeAlgorithm.RESIZE_LINEAR: 0>
    RESIZE_NEAREST: typing.ClassVar[ResizeAlgorithm]  # value = <ResizeAlgorithm.RESIZE_NEAREST: 2>
    __members__: typing.ClassVar[dict[str, ResizeAlgorithm]]  # value = {'RESIZE_LINEAR': <ResizeAlgorithm.RESIZE_LINEAR: 0>, 'RESIZE_CUBIC': <ResizeAlgorithm.RESIZE_CUBIC: 1>, 'RESIZE_NEAREST': <ResizeAlgorithm.RESIZE_NEAREST: 2>, 'RESIZE_BILINEAR_PILLOW': <ResizeAlgorithm.RESIZE_BILINEAR_PILLOW: 3>, 'RESIZE_BICUBIC_PILLOW': <ResizeAlgorithm.RESIZE_BICUBIC_PILLOW: 4>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
BGR: ColorFormat  # value = <ColorFormat.BGR: 6>
BGRX: ColorFormat  # value = <ColorFormat.BGRX: 9>
CONSTANT: PaddingMode  # value = <PaddingMode.CONSTANT: 0>
GRAY: ColorFormat  # value = <ColorFormat.GRAY: 7>
I420_SINGLE_PLANE: ColorFormat  # value = <ColorFormat.I420_SINGLE_PLANE: 3>
I420_THREE_PLANES: ColorFormat  # value = <ColorFormat.I420_THREE_PLANES: 4>
NV12_SINGLE_PLANE: ColorFormat  # value = <ColorFormat.NV12_SINGLE_PLANE: 1>
NV12_TWO_PLANES: ColorFormat  # value = <ColorFormat.NV12_TWO_PLANES: 2>
REFLECT: PaddingMode  # value = <PaddingMode.REFLECT: 2>
RESIZE_BICUBIC_PILLOW: ResizeAlgorithm  # value = <ResizeAlgorithm.RESIZE_BICUBIC_PILLOW: 4>
RESIZE_BILINEAR_PILLOW: ResizeAlgorithm  # value = <ResizeAlgorithm.RESIZE_BILINEAR_PILLOW: 3>
RESIZE_CUBIC: ResizeAlgorithm  # value = <ResizeAlgorithm.RESIZE_CUBIC: 1>
RESIZE_LINEAR: ResizeAlgorithm  # value = <ResizeAlgorithm.RESIZE_LINEAR: 0>
RESIZE_NEAREST: ResizeAlgorithm  # value = <ResizeAlgorithm.RESIZE_NEAREST: 2>
RGB: ColorFormat  # value = <ColorFormat.RGB: 5>
RGBX: ColorFormat  # value = <ColorFormat.RGBX: 8>
SYMMETRIC: PaddingMode  # value = <PaddingMode.SYMMETRIC: 3>
UNDEFINED: ColorFormat  # value = <ColorFormat.UNDEFINED: 0>
