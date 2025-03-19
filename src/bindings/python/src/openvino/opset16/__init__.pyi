from __future__ import annotations
from openvino.opset16.ops import identity
from openvino.opset16.ops import istft
from openvino.opset16.ops import segment_max
from . import ops
__all__ = ['identity', 'istft', 'ops', 'segment_max']
