from functools import partial
from typing import Any, Dict, List, Optional

from _pyngraph import NodeFactory as _NodeFactory
from ngraph.impl import Node

DEFAULT_OPSET = "opset3"


class NodeFactory(object):
    """Factory front-end to create node objects."""

    def __init__(self, opset_version: str = DEFAULT_OPSET) -> None:
        """Create the NodeFactory object.

        :param      opset_version:  The opset version the factory will use to produce ops from.
        """
        self.factory = _NodeFactory(opset_version)

    def create(
        self, op_type_name: str, arguments: List[Node], attributes: Optional[Dict[str, Any]] = None
    ) -> Node:
        """Create node object from provided description.

        :param      op_type_name:  The operator type name.
        :param      arguments:     The operator arguments.
        :param      attributes:    The operator attributes.

        :returns:   Node object representing requested operator with attributes set.
        """
        if attributes is None:
            attributes = {}
        node = self.factory.create(op_type_name, arguments, attributes)

        for atr_name in node._get_attributes().keys():
            setattr(node,
                    self._normalize_atr_name_getter(atr_name),
                    # partial(Node._get_attribute, node, atr_name))
                    partial(NodeFactory._get_node_attr_value, node, atr_name))
            setattr(node,
                    self._normalize_atr_name_setter(atr_name),
                    partial(Node._set_attribute, node, atr_name))
        return node

    def _normalize_atr_name_getter(self, attr_name: str) -> str:
        """Normalizes atr name to be suitable for getter function name.

        :param      attr_name:  The attribute name to normalize

        :returns:   The appropriate getter function name.
        """
        return "get_" + attr_name.replace(".", "_")

    def _normalize_atr_name_setter(self, attr_name: str) -> str:
        """Normalizes atr name to be suitable for setter function name.

        :param      attr_name:  The attribute name to normalize

        :returns:   The appropriate setter function name.
        """
        return "set_" + attr_name.replace(".", "_")

    @staticmethod
    def _get_node_attr_value(node: Node, attr_name: str) -> Any:
        """Get provided node attribute value.

        :param      node:       The node we retrieve attribute value from.
        :param      attr_name:  The attribute name.

        :returns:   The node attribute value.
        """
        return node._get_attributes()[attr_name]
