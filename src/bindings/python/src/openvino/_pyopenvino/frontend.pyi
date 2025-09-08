# type: ignore
from __future__ import annotations
import collections.abc
import openvino._pyopenvino
import typing
__all__ = ['OpExtension']
class OpExtension(openvino._pyopenvino._ConversionExtension):
    @typing.overload
    def __init__(self, fw_type_name: str, attr_names_map: collections.abc.Mapping[str, str] = {}, attr_values_map: collections.abc.Mapping[str, typing.Any] = {}) -> None:
        ...
    @typing.overload
    def __init__(self, ov_type_name: str, fw_type_name: str, attr_names_map: collections.abc.Mapping[str, str] = {}, attr_values_map: collections.abc.Mapping[str, typing.Any] = {}) -> None:
        ...
    @typing.overload
    def __init__(self, fw_type_name: str, in_names_vec: collections.abc.Sequence[str], out_names_vec: collections.abc.Sequence[str], attr_names_map: collections.abc.Mapping[str, str] = {}, attr_values_map: collections.abc.Mapping[str, typing.Any] = {}) -> None:
        ...
    @typing.overload
    def __init__(self, ov_type_name: str, fw_type_name: str, in_names_vec: collections.abc.Sequence[str], out_names_vec: collections.abc.Sequence[str], attr_names_map: collections.abc.Mapping[str, str] = {}, attr_values_map: collections.abc.Mapping[str, typing.Any] = {}) -> None:
        ...
