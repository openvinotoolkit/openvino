from __future__ import annotations
from openvino.utils.decorators import MultiMethod
from openvino.utils.decorators import _get_name
from openvino.utils.decorators import _set_node_friendly_name
from openvino.utils.decorators import binary_op
from openvino.utils.decorators import custom_preprocess_function
from openvino.utils.decorators import nameable_op
from openvino.utils.decorators import overloading
from openvino.utils.decorators import unary_op
__all__ = ['MultiMethod', 'binary_op', 'custom_preprocess_function', 'nameable_op', 'overloading', 'registry', 'unary_op']
registry: dict  # value = {'read_value': <openvino.utils.decorators.MultiMethod object>, 'constant': <openvino.utils.decorators.MultiMethod object>}
