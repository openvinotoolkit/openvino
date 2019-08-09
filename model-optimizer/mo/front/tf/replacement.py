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

from mo.front.common.custom_replacement_registry import CustomReplacementRegistry
from mo.front.common.replacement import FrontReplacementSubgraph, FrontReplacementPattern
from mo.front.subgraph_matcher import SubgraphMatcher, SubgraphMatch
from mo.front.tf.custom_subgraph_call import merge_nodes
from mo.graph.graph import Graph
from mo.ops.op import Op
from mo.utils import class_registration
from mo.utils.graph import is_connected_component
from mo.utils.replacement_pattern import ReplacementPattern


class FrontReplacementFromConfigFileGeneral(FrontReplacementPattern):
    """
    Translates graph to transform with the configuration files with custom attributes
    """
    replacement_id = ""

    def __init__(self):
        super().__init__()

    def transform_graph(self, graph, replacement_descriptions):
        raise Exception('Function "transform_graph" must be overridden in the sub-class')

    def find_and_replace_pattern(self, graph: Graph):
        replacement_descriptions = CustomReplacementRegistry().get_custom_replacement_description(self.replacement_id)
        if replacement_descriptions is None or len(replacement_descriptions) < 1:
            log.info("Failed to find custom replacement description with id '{}'".format(self.replacement_id))
            return
        for desc in replacement_descriptions:
            if 'custom_attributes' in desc._replacement_desc:
                self.transform_graph(graph, desc._replacement_desc['custom_attributes'])
            else:
                log.info("Failed to find \'custom_attributes\' in replacement description with id '{}'".format(
                    self.replacement_id))

    registered_ops = {}
    registered_cls = []

    @classmethod
    def class_type(cls):
        return class_registration.ClassType.FRONT_REPLACER


ReplacementPattern.excluded_replacers.append(FrontReplacementFromConfigFileGeneral)


class FrontReplacementFromConfigFileSubGraph(FrontReplacementSubgraph):
    """
    Replace sub-graph defined in the configuration files with a sub-graph of operations.
    """
    replacement_id = ""

    def __init__(self):
        super().__init__()

    def nodes_to_remove(self, graph: Graph, match: SubgraphMatch):
        return match.matched_nodes_names()

    def find_and_replace_pattern(self, graph: Graph):
        replacement_descriptions = CustomReplacementRegistry().get_custom_replacement_description(self.replacement_id)
        if replacement_descriptions is None:
            log.info("Failed to find custom replacement description with id '{}'".format(self.replacement_id))
            return
        # there are a list of custom replacements descriptions that have the same replacement id
        for replacement_description in replacement_descriptions:
            sub_graph_matcher = SubgraphMatcher(replacement_description)
            matched_instances = list(sub_graph_matcher.matched_sub_graph_instances(graph))
            if not len(matched_instances):
                log.error("Failed to match nodes from custom replacement description with id '{}':\nIt means model and "
                          "custom replacement description are incompatible.\nTry to correct custom replacement "
                          "description according to documentation with respect to model node names"
                          "".format(self.replacement_id))
            for match in matched_instances:
                if not is_connected_component(graph, match.matched_nodes_names()):
                    log.warning("The following nodes don't form connected sub-graph: {}".format(
                        match.matched_nodes_names()))
                    # graph.dump_graph_for_graphviz(match.matched_nodes_names())
                self.replace_sub_graph(graph, match)

    registered_ops = {}
    registered_cls = []

    @classmethod
    def class_type(cls):
        return class_registration.ClassType.FRONT_REPLACER


ReplacementPattern.excluded_replacers.append(FrontReplacementFromConfigFileSubGraph)


class FrontReplacementFromConfigFileOp(FrontReplacementFromConfigFileSubGraph):
    """
    Replace sub-graph defined in the configuration file with as single operation.
    """
    replacement_id = ""

    def __init__(self):
        super().__init__()

    def input_edges_match(self,  # pylint: disable=method-hidden
                          graph: Graph,
                          match: SubgraphMatch,
                          new_sub_graph: dict):
        """
        Function that generates matching of sub-graph input edges to a new sub-graph input edges. It works in case when
        the sub-graph is replaced with a single custom-layer node.
        :param graph: networkX graph to operate on.
        :param match: object describing matched sub-graph.
        :param new_sub_graph: dictionary of Nodes objects that forms new sub-graph.
        :return: object describing edges matching.
        """
        input_edges_match = dict()
        inputs_count = match.inputs_count()
        for sub_graph_input_port in range(inputs_count):
            # just create single edge for each input port of the sub-graph
            input_node, input_port = match.input_nodes(sub_graph_input_port)[0]
            input_edges_match[(input_node.id, input_port)] = (new_sub_graph['new_node'].id, sub_graph_input_port)
        return input_edges_match

    def output_edges_match(self,  # pylint: disable=method-hidden
                           graph: Graph,
                           match: SubgraphMatch,
                           new_sub_graph: dict):
        """
        Function that generates matching of sub-graph output edges to a new sub-graph output edges. It works in case
        when the sub-graph is replaced with a single custom-layer node.
        :param graph: networkX graph to operate on.
        :param match: object describing matched sub-graph.
        :param new_sub_graph: dictionary of Nodes objects that forms new sub-graph.
        :return: object describing edges matching.
        """
        output_edges_match = dict()
        outputs_count = match.outputs_count()
        # prepare output_edges_match based on custom replacement configuration file
        for sub_graph_output_port in range(outputs_count):
            output_node, output_port = match.output_node(sub_graph_output_port)
            output_edges_match[(output_node.id, output_port)] = (new_sub_graph['new_node'].id, sub_graph_output_port)
        return output_edges_match

    def generate_sub_graph(self, graph: Graph, match: SubgraphMatch):
        replacement_desc = match.custom_replacement_desc
        op = Op.get_op_class_by_name(replacement_desc.op)(graph, match.custom_replacement_desc.custom_attributes)
        op.default_backend_attrs = list(match.custom_replacement_desc.custom_attributes.keys())
        if 'infer' not in op.attrs:
            # update IE attrs
            op.substitute_ie_attrs(op.attrs)
            node = merge_nodes(graph, match.matched_nodes_names(), replacement_desc.get_inputs_description(),
                               replacement_desc.get_outputs_description())
            node.name = graph.unique_id(op.attrs['type'])
            node_attrs = graph.node[node.id]
            # copy attributes which are defined in the custom operation
            for key in op.attrs.keys():
                if key not in ['name', 'op']:
                    node_attrs[key] = op.attrs[key]
            # functions below should return nothing because 'merge_nodes' already created input/output edges
            self.input_edges_match = lambda gr, ma, new_sub_graph: dict()  # pylint: disable=method-hidden
            self.output_edges_match = lambda gr, ma, new_sub_graph: dict()  # pylint: disable=method-hidden
        else:
            node = op.add_node(name=op.attrs['type'] + '_')
            node.type = op.attrs['type']
        return {'new_node': node}

    registered_ops = {}
    registered_cls = []

    @classmethod
    def class_type(cls):
        return class_registration.ClassType.FRONT_REPLACER


ReplacementPattern.excluded_replacers.append(FrontReplacementFromConfigFileOp)
