# type: ignore
from __future__ import annotations
from . import device
from . import hint
from . import intel_auto
from . import intel_cpu
from . import intel_gpu
from . import log
from . import _properties
from . import streams
from openvino.properties._properties import __make_properties
from openvino._pyopenvino import properties as __properties
from openvino._pyopenvino.properties import CacheMode
from openvino._pyopenvino.properties import WorkloadType
__all__ = ['CacheMode', 'WorkloadType', 'device', 'hint', 'intel_auto', 'intel_cpu', 'intel_gpu', 'log', 'streams']
