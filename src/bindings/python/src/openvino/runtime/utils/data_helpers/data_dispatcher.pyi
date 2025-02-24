from __future__ import annotations
from openvino.utils.data_helpers.data_dispatcher import _data_dispatch
from openvino.utils.data_helpers.data_dispatcher import create_copied
from openvino.utils.data_helpers.data_dispatcher import create_shared
from openvino.utils.data_helpers.data_dispatcher import get_request_tensor
from openvino.utils.data_helpers.data_dispatcher import is_list_simple_type
from openvino.utils.data_helpers.data_dispatcher import normalize_arrays
from openvino.utils.data_helpers.data_dispatcher import set_request_tensor
from openvino.utils.data_helpers.data_dispatcher import to_c_style
from openvino.utils.data_helpers.data_dispatcher import update_inputs
from openvino.utils.data_helpers.data_dispatcher import update_tensor
from openvino.utils.data_helpers.data_dispatcher import value_to_tensor
import typing
__all__ = ['ContainerTypes', 'ScalarTypes', 'ValidKeys', 'create_copied', 'create_shared', 'get_request_tensor', 'is_list_simple_type', 'normalize_arrays', 'set_request_tensor', 'to_c_style', 'update_inputs', 'update_tensor', 'value_to_tensor']
ContainerTypes: typing._UnionGenericAlias  # value = typing.Union[dict, list, tuple, openvino.utils.data_helpers.wrappers.OVDict]
ScalarTypes: typing._UnionGenericAlias  # value = typing.Union[numpy.number, int, float]
ValidKeys: typing._UnionGenericAlias  # value = typing.Union[str, int, openvino._pyopenvino.ConstOutput]
