# type: ignore
"""
openvino.properties.intel_gpu.hint submodule that simulates ov::intel_gpu::hint
"""
from __future__ import annotations
import openvino._pyopenvino
import openvino._pyopenvino.properties.hint
import typing
__all__ = ['ThrottleLevel', 'available_device_mem', 'host_task_priority', 'queue_priority', 'queue_throttle']
class ThrottleLevel:
    """
    openvino.properties.intel_gpu.hint.ThrottleLevel that simulates ov::intel_gpu::hint::ThrottleLevel
    """
    DEFAULT: typing.ClassVar[openvino._pyopenvino.properties.hint.Priority]  # value = <Priority.MEDIUM: 1>
    HIGH: typing.ClassVar[openvino._pyopenvino.properties.hint.Priority]  # value = <Priority.HIGH: 2>
    LOW: typing.ClassVar[openvino._pyopenvino.properties.hint.Priority]  # value = <Priority.LOW: 0>
    MEDIUM: typing.ClassVar[openvino._pyopenvino.properties.hint.Priority]  # value = <Priority.MEDIUM: 1>
@typing.overload
def available_device_mem() -> str:
    ...
@typing.overload
def available_device_mem(arg0: int) -> tuple[str, openvino._pyopenvino.OVAny]:
    ...
@typing.overload
def host_task_priority() -> str:
    ...
@typing.overload
def host_task_priority(arg0: openvino._pyopenvino.properties.hint.Priority) -> tuple[str, openvino._pyopenvino.OVAny]:
    ...
@typing.overload
def queue_priority() -> str:
    ...
@typing.overload
def queue_priority(arg0: openvino._pyopenvino.properties.hint.Priority) -> tuple[str, openvino._pyopenvino.OVAny]:
    ...
@typing.overload
def queue_throttle() -> str:
    ...
@typing.overload
def queue_throttle(arg0: openvino._pyopenvino.properties.hint.Priority) -> tuple[str, openvino._pyopenvino.OVAny]:
    ...
