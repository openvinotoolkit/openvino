"""
 Copyright (c) 2017-2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import logging as log
import re

from mo.graph.graph import Node, Graph
from mo.utils.custom_replacement_config import CustomReplacementDescriptor
from mo.utils.error import Error
from mo.utils.graph import nodes_matching_name_pattern, sub_graph_between_nodes
from mo.utils.utils import refer_to_faq_msg


def find_object_by_pattern(names: list, pattern: str):
    """
    :param names: list of names to find objects from.
    :param pattern: regular expression for the name.
    :return: list of matched objects.
    """
    compiled_pattern = re.compile(pattern)
    return [name for name in names if re.match(compiled_pattern, name)]


class SubgraphMatch(object):
    """
    Class providing information about matched sub-graph.
    """

    def __init__(self, graph: Graph, replacement_desc: CustomReplacementDescriptor, matched_nodes: list,
                 inputs_order: list, outputs_order: list, prefix: str):
        """
        Creates instance of a SubgraphMatch class from the provided configuration.
        :param graph: networkx graph.
        :param replacement_desc: CustomReplacementDescriptor object describing sub-graph.
        :param matched_nodes: list of matched nodes.
        :param inputs_order: nodes description in the format described in the FrontReplacementFromConfigFileSubGraph.
        :param outputs_order: nodes description in the format described in the FrontReplacementFromConfigFileSubGraph.
        :param prefix: optional prefix of the node names. Is not used in the sub-graph match by points.
        """
        self._input_nodes_map = dict()
        self._output_nodes_map = dict()
        self._matched_nodes_names = matched_nodes
        self.graph = graph
        self.custom_replacement_desc = replacement_desc
        self.scope = prefix

        for sub_graph_input_port, input_desc in enumerate(inputs_order):
            for node_pattern, node_in_port in input_desc:
                node = self.node_by_pattern(node_pattern)
                if node is not None:
                    self._add_input_node(node.id, node_in_port, sub_graph_input_port)

        for sub_graph_output_port, (node_pattern, out_port) in enumerate(outputs_order):
            node = self.node_by_pattern(node_pattern)
            if node is not None:
                self._add_output_node(node.id, out_port, sub_graph_output_port)

    def matched_nodes_names(self):
        """
        Returns list of node names in the matched sub-graph.
        :return: list of node names in the matched sub-graph.
        """
        return self._matched_nodes_names

    def inputs_count(self):
        """
        Returns number of inputs for the matched sub-graph. Only unique input tensors are considered, thus if the same
        tensor is consumed by two or more input nodes of the sub-graph it is counted only once.
        :return: Number or unique input tensors.
        """
        return len(self._input_nodes_map.keys())

    def outputs_count(self):
        """
        Returns number of outputs for the matched sub-graph. Only unique output tensors are considered, thus if the same
        tensor is consumed by two or more nodes outside of the sub-graph it is counted only once.
        :return: Number or unique input tensors.
        """
        return len(self._output_nodes_map.keys())

    def input_nodes(self, port: int):
        """
        Returns list of tuples where the first element is a Node of the sub-graph and the second is the input port for
        that node. Each node of this list gets the same input tensor through the input port with number 'port' of the
        sub-graph.

        For example, if the returned list requested for port 'portSG' is the following: [(NodeA, portA), (nodeB, portB)]
        then the same tensor is passed to node 'NodeA' as input with number 'portA' and node 'nodeB' as input with
        number 'portB' for the sub-graph input with number 'portSG'.
        :param port: input port of the sub-graph.
        :return: list describing nodes of the sub-graph getting tensor through the specified port.
        """
        return self._input_nodes_map[port]

    def single_input_node(self, port: int):
        """
        The function does the same as function 'input_nodes' but it relies on fact that there is just one node that
        gets input tensor for sub-graph input with number 'port', so it return just tuple (Node, nodePort) or raises
        exception if the amount of nodes is not equal to 1.
        :param port: input port of the sub-graph.
        :return: tuple describing node of the sub-graph getting tensor through the specified port.
        """
        input_nodes = self.input_nodes(port)
        if len(input_nodes) != 1:
            raise Error('The amount of input nodes for port "{}" is not equal to 1. '.format(port) +
                        refer_to_faq_msg(33))
        return input_nodes[0]

    def output_node(self, port: int):
        """
        Returns a tuple where the first element is a Node of the sub-graph and the second is the output port of that
        node. Th node produces output tensor through the output port with number 'port' of the sub-graph.
        :param port: output port of the sub-graph.
        :return: tuple describing node of the sub-graph producing sub-graph output tensor through the specified port.
        """
        return self._output_nodes_map[port]

    def node_by_pattern(self, pattern: str):
        """
        Returns Node from the list of sub-graph nodes matching node name regular expression 'pattern'. If there are more
        than one nodes matched then the function raises exception.
        :param pattern: the regular expression for the node name.
        :return: matched Node.
        """
        if self.scope != '':
            if self.scope[-1] == '/':
                pattern = self.scope + pattern
            else:
                pattern = self.scope + '/' + pattern
        found_names = find_object_by_pattern(self._matched_nodes_names, pattern)
        if len(found_names) > 1:
            raise Error('The amount of nodes matched pattern "{}" is more than 1. '.format(pattern) +
                        refer_to_faq_msg(78))
        if len(found_names) == 0:
            return None
        return Node(self.graph, found_names[0])

    def _add_input_node(self, node_name: str, node_port: int, sub_graph_input_port: int):
        self._input_nodes_map.setdefault(sub_graph_input_port, []).append((Node(self.graph, node_name), node_port))

    def _add_output_node(self, node_name: str, node_port: int, sub_graph_output_port: int):
        if sub_graph_output_port in self._output_nodes_map:
            raise Error('Output node for port "{}" has already been specified. '.format(sub_graph_output_port) +
                        refer_to_faq_msg(34))
        self._output_nodes_map[sub_graph_output_port] = (Node(self.graph, node_name), node_port)


# TODO looks like this class is not needed. Can be implemented as pure functions.
class SubgraphMatcher(object):
    def __init__(self, replacement_descriptor: CustomReplacementDescriptor):
        self.replacement_desc = replacement_descriptor

    def _match_sub_graph_for_scope(self, graph: Graph, scope_pattern: str):
        """
        :param graph: networkx graph to find sub-graph in.
        :param scope_pattern: regular expression specifying sub-graph scope.
        :return: an object describing matched sub-graph.
        """
        inputs_order = self.replacement_desc.get_inputs_description()
        outputs_order = self.replacement_desc.get_outputs_description()

        for list_nodes in inputs_order:
            for node_name_pattern, port in list_nodes:
                if len(find_object_by_pattern(graph.nodes(), '.*' + node_name_pattern)) == 0:
                    log.info('Node "{} does not exist in the graph". Failed to match sub-graph by scope "{}".'.format(
                        node_name_pattern, self.replacement_desc.id))
                    return None

        matched_nodes = nodes_matching_name_pattern(graph, scope_pattern)
        if len(matched_nodes) == 0:
            log.info('There are no instances of the sub-graph by scope "{}"'.format(scope_pattern))
            return None

        return SubgraphMatch(graph, self.replacement_desc, matched_nodes, inputs_order, outputs_order, scope_pattern)

    def _match_sub_graph_for_points(self, graph: Graph):
        """
        :param graph: networkx graph to find sub-graph in.
        :return: an object describing matched sub-graph.
        """
        start_points = self.replacement_desc.get_internal_input_nodes(graph)
        end_points = self.replacement_desc.get_internal_output_nodes(graph)
        # check that start and end points exist in the graph
        for node_name in start_points + end_points:
            if node_name not in graph.nodes():
                log.info('Node "{}" does not exist in the graph. Failed to match sub-graph by points "{}".'.format(
                    node_name, self.replacement_desc.id))
                return None

        matched_nodes = sub_graph_between_nodes(graph, start_points, end_points)
        return SubgraphMatch(graph, self.replacement_desc, matched_nodes,
                             self.replacement_desc.get_inputs_description(),
                             self.replacement_desc.get_outputs_description(), '')

    def matched_sub_graph_instances(self, graph: Graph):
        """
        Generator to product all instances of matched sub-graphs.
        :param graph: graph to find instances in.
        :return: generator producing SubGraphMatch objects.
        """
        if self.replacement_desc.match_kind == 'points':  # instance is specified with lists of start/end nodes
            match = self._match_sub_graph_for_points(graph)
            if match is not None:
                yield match
        elif self.replacement_desc.match_kind == 'scope':  # instance is specified with a node name pattern
            for instance in self.replacement_desc.sub_graph_instances():
                match = self._match_sub_graph_for_scope(graph, instance)
                if match is not None:
                    yield match
        else:
            raise Error('Unsupported match kind "{}". Match kinds "points" or "scope" are supported only. '.format(
                self.replacement_desc.match_kind) +
                        refer_to_faq_msg(35))
