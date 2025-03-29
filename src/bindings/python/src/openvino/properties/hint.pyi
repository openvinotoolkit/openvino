# type: ignore
from __future__ import annotations
from openvino.properties._properties import __make_properties
from openvino._pyopenvino.properties.hint import ExecutionMode
from openvino._pyopenvino.properties.hint import ModelDistributionPolicy
from openvino._pyopenvino.properties.hint import PerformanceMode
from openvino._pyopenvino.properties.hint import Priority
from openvino._pyopenvino.properties.hint import SchedulingCoreType
from openvino._pyopenvino.properties import hint as __hint
__all__ = ['ExecutionMode', 'ModelDistributionPolicy', 'PerformanceMode', 'Priority', 'SchedulingCoreType']
