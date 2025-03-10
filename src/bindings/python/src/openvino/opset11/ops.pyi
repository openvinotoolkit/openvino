import openvino._pyopenvino
import typing
from functools import partial as partial
from openvino._pyopenvino import Node as Node
from openvino.utils.decorators import nameable_op as nameable_op
from openvino.utils.types import as_nodes as as_nodes

def interpolate(*args, **kwargs) -> openvino._pyopenvino.Node: ...
def topk(*args, **kwargs) -> openvino._pyopenvino.Node: ...

NodeInput: typing._UnionGenericAlias
