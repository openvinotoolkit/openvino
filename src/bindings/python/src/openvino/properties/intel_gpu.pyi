# type: ignore
from __future__ import annotations
from openvino.properties._properties import __make_properties
from openvino._pyopenvino.properties import intel_gpu as __intel_gpu
from openvino._pyopenvino.properties.intel_gpu import CapabilityGPU
from openvino._pyopenvino.properties.intel_gpu import MemoryType
__all__ = ['CapabilityGPU', 'MemoryType']
