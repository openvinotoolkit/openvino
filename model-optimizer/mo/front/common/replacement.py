"""
 Copyright (c) 2017-2018 Intel Corporation

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

import networkx as nx

from mo.front.subgraph_matcher import SubgraphMatch
from mo.graph.graph import Node, merge_edge_props, get_sorted_inputs
from mo.middle.pattern_match import apply_pattern
from mo.utils import class_registration
from mo.utils.replacement_pattern import ReplacementPattern


class FrontReplacementPattern(ReplacementPattern):
    registered_ops = {}
    registered_cls = []

    def pattern(self):
        raise Exception('Function "pattern" must be overridden in the sub-class')

    @classmethod
    def class_type(cls):
        return class_registration.ClassType.FRONT_REPLACER


ReplacementPattern.excluded_replacers.append(FrontReplacementPattern)


class FrontReplacementSubgraph(FrontReplacementPattern):
    """
    Replace pattern defined set of nodes with a sub-graph.
    """
    replacement_id = 'None'

    def __init__(self):
        pass

    @staticmethod
    def extract_port(node_port):
        return node_port if isinstance(node_port, tuple) else (node_port, 0)

    @staticmethod
    def replace_input_edges(graph: nx.DiGraph, input_edges_match: dict):
        """
        Replacing existing input/output edges with a new ones to a new sub-graph.
        :param graph: networkX graph to operate on.
        :param input_edges_match: match of input edges between old and new sub-graph.
        :return: None
        """
        for old_name_port, new_name_port in input_edges_match.items():
            old_node_name, old_in_port = __class__.extract_port(old_name_port)
            new_node_name, new_in_port = __class__.extract_port(new_name_port)
            old_node = Node(graph, old_node_name)
            src_node_name = get_sorted_inputs(old_node)[old_in_port][0]
            edge_attrs = graph[src_node_name][old_node_name][0].copy()
            edge_attrs['in'] = new_in_port
            graph.add_edge(src_node_name, new_node_name, **edge_attrs)
            log.debug("Created edge from {} to {} with attrs: {}".format(src_node_name, new_node_name, edge_attrs))

    @staticmethod
    def replace_output_edges(graph: nx.DiGraph, output_edges_match: dict):
        """
        Replacing existing input/output edges with a new ones to a new sub-graph.
        :param graph: networkX graph to operate on.
        :param output_edges_match: match of output edges between old and new sub-graph.
        :return: None
        """
        for old_name_port, new_name_port in output_edges_match.items():
            old_node_name, old_out_port = __class__.extract_port(old_name_port)
            new_node_name, new_out_port = __class__.extract_port(new_name_port)
            for src, dst, edge_attrs in graph.out_edges(old_node_name, data=True):
                if edge_attrs['out'] == old_out_port:
                    new_edge_attrs = edge_attrs.copy()
                    new_edge_attrs['out'] = new_out_port
                    graph.add_edge(new_node_name, dst, **new_edge_attrs)
                    log.debug("Created edge from {} to {} with attrs: {}".format(new_node_name, dst, new_edge_attrs))

    def input_edges_match(self, graph: nx.MultiDiGraph, match: object, new_sub_graph: dict):
        """
        Default implementation doesn't add new input edges automatically.
        """
        return {}

    def output_edges_match(self, graph: nx.MultiDiGraph, match: object, new_sub_graph: dict):
        """
        Default implementation doesn't add new output edges automatically.
        """
        return {}

    def generate_sub_graph(self, graph: nx.MultiDiGraph, match: object):
        raise Exception("The function 'generate_sub_graph' must be implemented in the sub-class.")

    def nodes_to_remove(self, graph: nx.MultiDiGraph, match: dict):
        """
        Default implementation generates list of all matched nodes. So all matched nodes will be removed.
        """
        return [node.id for node in match.values()]

    def replace_sub_graph(self, graph: nx.MultiDiGraph, match: [dict, SubgraphMatch]):
        log.debug('replace_sub_graph: "{}" matched nodes: {}'.format(self.replacement_id,
                                                                     '\n'.join(sorted(match.matched_nodes_names()))))
        new_sub_graph = self.generate_sub_graph(graph, match)  # pylint: disable=assignment-from-no-return
        self.replace_input_edges(graph, self.input_edges_match(graph, match, new_sub_graph))
        self.replace_output_edges(graph, self.output_edges_match(graph, match, new_sub_graph))

        remove_nodes = self.nodes_to_remove(graph, match)
        log.debug(
            'replace_sub_graph: "{}" removing nodes: {}'.format(self.replacement_id, '\n'.join(sorted(remove_nodes))))
        graph.remove_nodes_from(remove_nodes)

    def find_and_replace_pattern(self, graph: nx.MultiDiGraph):
        apply_pattern(graph, action=self.replace_sub_graph, **self.pattern())

    registered_ops = {}
    registered_cls = []

    @classmethod
    def class_type(cls):
        return class_registration.ClassType.FRONT_REPLACER


ReplacementPattern.excluded_replacers.append(FrontReplacementSubgraph)


class FrontReplacementOp(FrontReplacementSubgraph):
    """
    A super class for an operation replacement.
    Replaces a single operation (identified by 'op' attribute) by a sub-graph of operations.
    It is a convenient specialization of FrontReplacementPattern.
    """
    op = 'UnknownOp'

    def pattern(self):
        return dict(
            nodes=[
                ('op', dict(op=self.__class__.op))],
            edges=[]
        )

    def replace_op(self, graph: nx.MultiDiGraph, node: Node):
        raise Exception("The function 'replace_op' must be implemented in the sub-class.")

    @staticmethod
    def gen_output_edges_match(node: Node, out_node_replace: list):
        out_edges_match_dict = dict()
        for old_out_port, new_node_desc in enumerate(out_node_replace):
            new_out_port = 0
            if new_node_desc is tuple:
                new_node_name = new_node_desc[0]
                new_out_port = new_node_desc[1]
            else:
                new_node_name = new_node_desc
            out_edges_match_dict[(node.id, old_out_port)] = (new_node_name, new_out_port)
        return out_edges_match_dict

    @staticmethod
    def update_input_edges_attrs(graph: nx.MultiDiGraph, node: Node, added_nodes: list):
        """
        Copy edge attributes from 'old' input edges of node 'node' to new input sub-graph edges.
        :param graph: graph to operate on
        :param node: Node object that was replaced.
        :param added_nodes: list of nodes names added.
        :return: None
        """
        for old_u, old_v, old_edge_attrs in graph.in_edges(node.id, data=True):
            for new_u, new_v, new_edge_attrs in graph.in_edges(added_nodes, data=True):
                if new_u not in added_nodes:  # external input to the sub-graph
                    if old_u == new_u and old_edge_attrs['out'] == new_edge_attrs['out']:
                        merge_edge_props(new_edge_attrs, old_edge_attrs)  # copy old edge attributes

    def replace_sub_graph(self, graph: nx.MultiDiGraph, match: dict):
        assert 'op' in match
        assert len(match) == 1
        node = match['op']
        nodes_before_replacement = graph.nodes()
        self.replace_output_edges(graph, self.gen_output_edges_match(node, self.replace_op(graph, node)))

        # nodes added by the 'replace_op' function call
        added_nodes = list(set(graph.nodes()) - set(nodes_before_replacement))
        self.update_input_edges_attrs(graph, node, added_nodes)

        # TODO Need to check if there are other users for these nodes
        remove_nodes = self.nodes_to_remove(graph, match)
        log.debug("Removing nodes: {}".format(remove_nodes))
        graph.remove_nodes_from(remove_nodes)

    registered_ops = {}
    registered_cls = []

    @classmethod
    def class_type(cls):
        return class_registration.ClassType.FRONT_REPLACER


ReplacementPattern.excluded_replacers.append(FrontReplacementOp)
