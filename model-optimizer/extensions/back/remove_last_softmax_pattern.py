"""
 Copyright (C) 2018-2020 Intel Corporation

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

from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph
from mo.middle.passes.eliminate import remove_op_node_with_data_node


class RemoveLastSoftMaxPattern(BackReplacementPattern):
    enabled = True
    graph_condition = [lambda graph: graph.graph['fw'] == 'kaldi' and graph.graph['cmd_params'].remove_output_softmax]

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('softmax_node', dict(op='SoftMax')),
                ('softmax_data', dict(kind='data')),
                ('op_output', dict(op='Result'))
            ],
            edges=[
                ('softmax_node', 'softmax_data'),
                ('softmax_data', 'op_output')
            ]
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        """
        Removes output SoftMax layer
        :param graph: graph to operate on
        :param match: dictionary with matched nodes
        """
        if len(match['softmax_data'].out_nodes()) == 1:
            remove_op_node_with_data_node(graph, match['softmax_node'])
        else:
            log.error("SoftMax is not last layer, so can't be removed", extra={'is_warning': True})


class RemoveLastLogSoftMaxPattern(BackReplacementPattern):
    enabled = True
    graph_condition = [lambda graph: graph.graph['fw'] == 'kaldi' and graph.graph['cmd_params'].remove_output_softmax]

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('softmax_node', dict(op='SoftMax')),
                ('softmax_data', dict(kind='data')),
                ('log_node', dict(op='Log')),
                ('log_data', dict(kind='data')),
                ('op_output', dict(op='Result'))
            ],
            edges=[
                ('softmax_node', 'softmax_data'),
                ('softmax_data', 'log_node'),
                ('log_node', 'log_data'),
                ('log_data', 'op_output')
            ]
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        """
        Removes output LogSoftMax layer
        :param graph: graph to operate on
        :param match: dictionary with matched nodes
        """
        if len(match['softmax_data'].out_nodes()) == 1 and len(match['log_data'].out_nodes()) == 1:
            remove_op_node_with_data_node(graph, match['log_node'])
            remove_op_node_with_data_node(graph, match['softmax_node'])
