# type: ignore
"""
openvino.properties.intel_cpu submodule that simulates ov::intel_cpu
"""
from __future__ import annotations
import openvino._pyopenvino
import typing
__all__ = ['denormals_optimization', 'sparse_weights_decompression_rate']
@typing.overload
def denormals_optimization() -> str:
    ...
@typing.overload
def denormals_optimization(arg0: bool) -> tuple[str, openvino._pyopenvino.OVAny]:
    ...
@typing.overload
def sparse_weights_decompression_rate() -> str:
    ...
@typing.overload
def sparse_weights_decompression_rate(arg0: float) -> tuple[str, openvino._pyopenvino.OVAny]:
    ...
