"""
 Copyright (c) 2018-2019 Intel Corporation

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
import numpy as np

from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Node, Graph
from mo.middle.passes.convert_data_type import data_type_str_to_np
from mo.ops.const import Const
from mo.utils.error import Error
from mo.utils.utils import refer_to_faq_msg


class CreateConstNodesReplacement(BackReplacementPattern):
    enabled = False

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('data', dict(kind='data'))
            ],
            edges=[]
        )

    @staticmethod
    def _check_bin_attrs(node):
        """Check that at least one output edge from node without 'bin' attribute."""
        out_edges = node.out_edges()
        bin_in_out_ports = ['bin' in edge for edge in out_edges]
        out_node = [node.has('op') and node.op == 'Result' for node in node.out_nodes()]
        return np.any(out_node) or not np.all(bin_in_out_ports)

    @staticmethod
    def _check_that_node_from_body(node):
        """Check that all output edges from node have 'internal_port_id'
        (that shows that this node is from TI body)"""
        n_ports = len(node.out_edges())
        internal_port_in_out_ports = ['internal_port_id' in edge for edge in node.out_edges()]
        return np.any(internal_port_in_out_ports) and n_ports

    def replace_pattern(self, graph: Graph, match: dict):
        """
            Adds layers with type 'Const' that produce blob from 'bin' file. The pass finds data nodes with one output which
            doesn't have edge with 'bin' attribute (or with two outputs and at least one output havent 'bin' attr)
            and generate Const op node before the node and data node before the Const node. The data node before 'Const'
            node is needed because the op node dumps input tensors to bin file.
        """
        node = match['data']
        if len(node.in_nodes()) > 0:
            return

        if self._check_bin_attrs(node):
            if node.has_valid('value'):
                const_node_name = graph.unique_id(node.id + '_const')
                log.debug("Added Const node '{}'".format(const_node_name))
                const_node = Const(graph, {'name': const_node_name, 'value': node.value}).create_node()
                const_node.add_input_port(0)
                graph.add_edges_from([(const_node_name, node.id, {'out': 0})])

                copy_data_node_name = graph.unique_id(node.id + '_copy_')
                graph.add_node(copy_data_node_name, kind='data', shape=node.shape, value=node.value)

                if node.has_valid('force_precision'):
                    Node(graph, copy_data_node_name)['force_precision'] = node.force_precision
                    const_node['force_precision'] = node.force_precision
                    const_node.out_port(0).set_data_type(data_type_str_to_np(const_node['force_precision']))
                else:
                    const_node.type_infer(const_node)
                graph.add_edges_from([(copy_data_node_name, const_node_name, {'in': 0, 'bin': 'custom'})])
            elif not self._check_that_node_from_body(node):
                log.debug('node = {}'.format(node.graph.node[node.id]))
                raise Error(
                    'Discovered data node without inputs and value, node.name = {}, consumer.name = {}. ' +
                    refer_to_faq_msg(23),
                    node.soft_get('name'),
                    node.out_node().soft_get('name') if len(node.out_nodes()) else "<no consumer>"
                )
