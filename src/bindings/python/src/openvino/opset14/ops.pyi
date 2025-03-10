import openvino._pyopenvino
import typing
from functools import partial as partial
from openvino._pyopenvino import Node as Node, Type as Type
from openvino.utils.decorators import nameable_op as nameable_op
from openvino.utils.types import as_node as as_node, as_nodes as as_nodes

def avg_pool(*args, **kwargs) -> openvino._pyopenvino.Node: ...
def convert_promote_types(*args, **kwargs) -> openvino._pyopenvino.Node: ...
def inverse(*args, **kwargs) -> openvino._pyopenvino.Node: ...
def max_pool(*args, **kwargs) -> openvino._pyopenvino.Node: ...

NodeInput: typing._UnionGenericAlias
TensorShape: typing._GenericAlias
