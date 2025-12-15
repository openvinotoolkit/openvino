# type: ignore
from __future__ import annotations
from openvino._pyopenvino.properties import intel_cpu as __intel_cpu
from openvino._pyopenvino.properties.intel_cpu import TbbPartitioner
from openvino.properties._properties import __make_properties
__all__: list[str] = ['TbbPartitioner']
