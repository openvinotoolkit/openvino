# type: ignore
from __future__ import annotations
import openvino._pyopenvino
import typing
__all__ = ['OpExtension']
class OpExtension(openvino._pyopenvino._ConversionExtension):
    @typing.overload
    def __init__(self, fw_type_name: str, attr_names_map: dict[str, str] = {}, attr_values_map: dict[str, typing.Any] = {}) -> None:
        ...
    @typing.overload
    def __init__(self, ov_type_name: str, fw_type_name: str, attr_names_map: dict[str, str] = {}, attr_values_map: dict[str, typing.Any] = {}) -> None:
        ...
    @typing.overload
    def __init__(self, fw_type_name: str, in_names_vec: list[str], out_names_vec: list[str], attr_names_map: dict[str, str] = {}, attr_values_map: dict[str, typing.Any] = {}) -> None:
        ...
    @typing.overload
    def __init__(self, ov_type_name: str, fw_type_name: str, in_names_vec: list[str], out_names_vec: list[str], attr_names_map: dict[str, str] = {}, attr_values_map: dict[str, typing.Any] = {}) -> None:
        ...
