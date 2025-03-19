# type: ignore
from functools import singledispatchmethod
from __future__ import annotations
from openvino.exceptions import UserInputError
from openvino._pyopenvino import Extension
from openvino._pyopenvino import Node
from openvino._pyopenvino import NodeFactory as _NodeFactory
from openvino._pyopenvino import Output
from pathlib import Path
from typing import Any
import openvino._pyopenvino
__all__ = ['Any', 'DEFAULT_OPSET', 'Extension', 'Node', 'NodeFactory', 'Output', 'Path', 'UserInputError', 'singledispatchmethod']
class NodeFactory:
    """
    Factory front-end to create node objects.
    """
    @staticmethod
    def _arguments_as_outputs(arguments: typing.List[typing.Union[openvino._pyopenvino.Node, openvino._pyopenvino.Output]]) -> typing.List[openvino._pyopenvino.Output]:
        ...
    @staticmethod
    def add_extension(*args, **kwargs) -> None:
        ...
    def _(self, extension: typing.Union[openvino._pyopenvino.Extension, typing.List[openvino._pyopenvino.Extension]]) -> None:
        """
        Add custom operations from extension library.
        
                Extends operation types available for creation by operations
                loaded from prebuilt C++ library. Enables instantiation of custom
                operations exposed in that library without direct use of
                operation classes. Other types of extensions, e.g. conversion
                extensions, if they are exposed in the library, are ignored.
        
                In case if an extension operation type from a library match
                one of existing operations registered before (from the standard
                OpenVINO opset or from another extension loaded earlier), a new
                operation overrides an old operation.
        
                Version of an operation is ignored: an operation with a given type and
                a given version/opset will override operation with the same type but
                different version/opset in the same NodeFactory instance.
                Use separate libraries and NodeFactory instances to differentiate
                versions/opsets.
        
                :param      extension:  A single Extension or list of Extensions.
                
        """
    def __init__(self, opset_version: str = 'opset13') -> None:
        """
        Create the NodeFactory object.
        
                :param      opset_version:  The opset version the factory will use to produce ops from.
                
        """
    def create(self, op_type_name: str, arguments: typing.Optional[typing.List[typing.Union[openvino._pyopenvino.Node, openvino._pyopenvino.Output]]] = None, attributes: typing.Optional[typing.Dict[str, typing.Any]] = None) -> openvino._pyopenvino.Node:
        """
        Create node object from provided description.
        
                The user does not have to provide all node's attributes, but only required ones.
        
                :param      op_type_name:  The operator type name.
                :param      arguments:     The operator arguments.
                :param      attributes:    The operator attributes.
        
                :return:   Node object representing requested operator with attributes set.
                
        """
def _get_node_factory(opset_version: typing.Optional[str] = None) -> NodeFactory:
    """
    Return NodeFactory configured to create operators from specified opset version.
    """
DEFAULT_OPSET: str = 'opset13'
