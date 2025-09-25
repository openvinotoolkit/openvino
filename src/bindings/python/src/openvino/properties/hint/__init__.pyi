# type: ignore
from __future__ import annotations
from openvino._pyopenvino.properties import hint as __hint
from openvino._pyopenvino.properties.hint import ExecutionMode
from openvino._pyopenvino.properties.hint import ModelDistributionPolicy
from openvino._pyopenvino.properties.hint import PerformanceMode
from openvino._pyopenvino.properties.hint import Priority
from openvino._pyopenvino.properties.hint import SchedulingCoreType
from openvino.properties._properties import __make_properties
__all__: list[str] = ['ExecutionMode', 'ModelDistributionPolicy', 'PerformanceMode', 'Priority', 'SchedulingCoreType']
