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

        The user does not have to provide all node's attributes, but only required ones.

        :param      op_type_name:  The operator type name.
        :param      arguments:     The operator arguments.
        :param      attributes:    The operator attributes.

        :returns:   Node object representing requested operator with attributes set.
        """
        if attributes is None:
            attributes = {}
        node = self.factory.create(op_type_name, arguments, attributes)

        # Currently we don't support any attribute getters & setters for TensorIterator node.
        if node.get_type_name() == "TensorIterator":
            return node

        # Set getters and setters for each node's attribute.
        #   node.get_attribute_name()
        #   node.set_attribute_name()
        # For compound (with more than one level of nesting) attributes of form ie.:
        # node.struct_member_name.some_metric.attr_name:
        #   node.get_some_metric_attr_name()
        #   node.set_some_metric_attr_name()
        all_attributes = node._get_attributes()
        for attr_name in all_attributes.keys():
            setattr(node,
                    self._normalize_attr_name_getter(attr_name),
                    partial(NodeFactory._get_node_attr_value, node, attr_name))
            setattr(node,
                    self._normalize_attr_name_setter(attr_name),
                    partial(NodeFactory._set_node_attr_value, node, attr_name))

        # Setup helper members for caching attribute values.
        # The cache would be lazily populated at first access attempt.
        setattr(node, "_attr_cache", dict())
        setattr(node, "_attr_cache_valid", bool(False))

        return node

    @staticmethod
    def _normalize_attr_name(attr_name: str, prefix: str) -> str:
        """Normalizes attribute name.

        :param      attr_name:  The attribute name.
        :param      prefix:     The prefix to attach to attribute name.

        :returns:   The modified attribute name.
        """
        # Trim first part of the name if there is only one level of attribute hierarchy.
        if attr_name.count(".") == 1:
            attr_name = attr_name[attr_name.find(".") + 1:]
        return prefix + attr_name.replace(".", "_")

    @classmethod
    def _normalize_attr_name_getter(cls, attr_name: str) -> str:
        """Normalizes atr name to be suitable for getter function name.

        :param      attr_name:  The attribute name to normalize

        :returns:   The appropriate getter function name.
        """
        return cls._normalize_attr_name(attr_name, "get_")

    @classmethod
    def _normalize_attr_name_setter(cls, attr_name: str) -> str:
        """Normalizes atr name to be suitable for setter function name.

        :param      attr_name:  The attribute name to normalize

        :returns:   The appropriate setter function name.
        """
        return cls._normalize_attr_name(attr_name, "set_")

    @staticmethod
    def _get_node_attr_value(node: Node, attr_name: str) -> Any:
        """Gets provided node attribute value.

        :param      node:       The node we retrieve attribute value from.
        :param      attr_name:  The attribute name.

        :returns:   The node attribute value.
        """
        if not node._attr_cache_valid:
            node._attr_cache = node._get_attributes()
            node._attr_cache_valid = True
        return node._attr_cache[attr_name]

    @staticmethod
    def _set_node_attr_value(node: Node, attr_name: str, value: Any) -> None:
        """Sets the node attribute value.

        :param      node:       The node we change attribute value for.
        :param      attr_name:  The attribute name.
        :param      value:      The new attribute value.
        """
        node._set_attribute(attr_name, value)
        node._attr_cache[attr_name] = value
