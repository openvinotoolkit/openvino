# type: ignore
"""
openvino.properties.intel_gpu submodule that simulates ov::intel_gpu
"""
from __future__ import annotations
from . import hint
import openvino._pyopenvino
import typing
__all__ = ['CapabilityGPU', 'MemoryType', 'device_total_mem_size', 'disable_winograd_convolution', 'enable_loop_unrolling', 'execution_units_count', 'hint', 'memory_statistics', 'uarch_version']
class CapabilityGPU:
    """
    openvino.properties.intel_gpu.CapabilityGPU submodule that simulates ov::intel_gpu::capability
    """
    HW_MATMUL: typing.ClassVar[str] = 'GPU_HW_MATMUL'
class MemoryType:
    """
    openvino.properties.intel_gpu.MemoryType submodule that simulates ov::intel_gpu::memory_type
    """
    buffer: typing.ClassVar[str] = 'GPU_BUFFER'
    surface: typing.ClassVar[str] = 'GPU_SURFACE'
def device_total_mem_size() -> str:
    ...
@typing.overload
def disable_winograd_convolution() -> str:
    ...
@typing.overload
def disable_winograd_convolution(arg0: bool) -> tuple[str, openvino._pyopenvino.OVAny]:
    ...
@typing.overload
def enable_loop_unrolling() -> str:
    ...
@typing.overload
def enable_loop_unrolling(arg0: bool) -> tuple[str, openvino._pyopenvino.OVAny]:
    ...
def execution_units_count() -> str:
    ...
def memory_statistics() -> str:
    ...
def uarch_version() -> str:
    ...
