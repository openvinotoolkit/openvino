"""
Generic utilities. Factor related functions out to separate files.
"""
from __future__ import annotations
from openvino._pyopenvino.util import numpy_to_c
from openvino._pyopenvino.util import replace_node
from openvino._pyopenvino.util import replace_output_update_name
from . import broadcasting
from . import data_helpers
from . import decorators
from . import input_validation
from . import node_factory
from . import reduction
from . import types
__all__ = ['broadcasting', 'data_helpers', 'decorators', 'input_validation', 'node_factory', 'numpy_to_c', 'reduction', 'replace_node', 'replace_output_update_name', 'types']
