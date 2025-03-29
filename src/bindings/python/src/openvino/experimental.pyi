# type: ignore
"""

Package: openvino
This module provides access to experimental functionality that is subject to change without prior notice.
"""
from __future__ import annotations
from openvino._pyopenvino.experimental import evaluate_as_partial_shape
from openvino._pyopenvino.experimental import evaluate_both_bounds
from openvino._pyopenvino.experimental import set_element_type
from openvino._pyopenvino.experimental import set_tensor_type
__all__ = ['evaluate_as_partial_shape', 'evaluate_both_bounds', 'set_element_type', 'set_tensor_type']
