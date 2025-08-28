# type: ignore
from __future__ import annotations
import collections.abc
import openvino._pyopenvino
import typing
__all__ = ['ConversionExtensionTensorflow', 'OpExtensionTensorflow']
class ConversionExtensionTensorflow(_ConversionExtensionTensorflow):
    def __init__(self, arg0: str, arg1: collections.abc.Callable[[openvino._pyopenvino.NodeContext], list[openvino._pyopenvino.Output]]) -> None:
        ...
class OpExtensionTensorflow(_ConversionExtensionTensorflow):
    @typing.overload
    def __init__(self, fw_type_name: str, attr_names_map: collections.abc.Mapping[str, str] = {}, attr_values_map: collections.abc.Mapping[str, typing.Any] = {}) -> None:
        ...
    @typing.overload
    def __init__(self, ov_type_name: str, fw_type_name: str, attr_names_map: collections.abc.Mapping[str, str] = {}, attr_values_map: collections.abc.Mapping[str, typing.Any] = {}) -> None:
        ...
class _ConversionExtensionTensorflow(openvino._pyopenvino.ConversionExtensionBase):
    pass
class _FrontEndDecoderBase(openvino._pyopenvino._IDecoder):
    def __init__(self) -> None:
        ...
class _FrontEndPyGraphIterator:
    def __init__(self) -> None:
        ...
