# type: ignore
from . import _properties
from . import device
from . import hint
from . import intel_auto
from . import intel_cpu
from . import intel_gpu
from . import log
from . import streams
from __future__ import annotations
from openvino._pyopenvino import properties as __properties
from openvino._pyopenvino.properties import CacheMode
from openvino._pyopenvino.properties import WorkloadType
from openvino.properties._properties import __make_properties
__all__: list[str] = ['CacheMode', 'WorkloadType', 'device', 'hint', 'intel_auto', 'intel_cpu', 'intel_gpu', 'log', 'streams']
