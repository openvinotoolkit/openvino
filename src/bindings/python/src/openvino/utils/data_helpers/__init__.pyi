# type: ignore
from __future__ import annotations
from . import data_dispatcher
from . import wrappers
from openvino.utils.data_helpers.data_dispatcher import _data_dispatch
from openvino.utils.data_helpers.wrappers import _InferRequestWrapper
from openvino.utils.data_helpers.wrappers import OVDict
from openvino.utils.data_helpers.wrappers import tensor_from_file
__all__ = ['OVDict', 'data_dispatcher', 'tensor_from_file', 'wrappers']
