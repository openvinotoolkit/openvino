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
from extensions.middle.TensorIteratorMerge import TensorIteratorMerge
from mo.graph.graph import Graph
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.result import Result


class AddFakeOutputsToSplit(MiddleReplacementPattern):
    """
        Adding fake outputs for Split nodes in case when it has less output ports than split parts:
        This pass:
            1. Looking for Split operations
            2. Check that Split have less connected output ports than split parts
            3. For every missed port adding this port, Output operation to this port
    """

    enabled = True

    def run_after(self):
        return [TensorIteratorMerge]

    @staticmethod
    def pattern():
        return dict(
            nodes=[('op', dict(kind='op', op='Split'))],
            edges=[],
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        node = match['op']

        if node.has_valid('out_ports_count') and len(node.out_edges()) < node.out_ports_count:
            for p in range(node.out_ports_count):
                if p not in node.out_ports():
                    node.add_output_port(p)
                if node.out_port(p).disconnected():
                    res_node = Result(graph, {'name': node.name + '/Fake_output_{}/'.format(p)}).create_node()
                    node.out_port(p).connect(res_node.in_port(0))


class AddFakeOutputsToVariadicSplit(MiddleReplacementPattern):
    """
        Adding fake outputs for VariadicSplit nodes in case when it has less output ports than split parts:
        This pass:
            1. Looking for VariadicSplit operations
            2. Check that VariadicSplit have less connected output ports than split parts
            3. For every missed port adding this port, Output operation to this port
    """

    enabled = True

    def run_after(self):
        return [TensorIteratorMerge]

    @staticmethod
    def pattern():
        return dict(
            nodes=[('op', dict(kind='op', op='VariadicSplit'))],
            edges=[],
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        node = match['op']
        axis = node.in_port(1).data.get_value()
        size_splits = node.in_port(2).data.get_value()

        output_shape = sum([node.out_node(port).shape[axis] for port in node.out_nodes()])

        if output_shape == node.in_port(0).data.get_shape()[axis]:
            return

        if not node.has_valid('out_ports_count'):
            node['out_ports_count'] = len(size_splits)

        AddFakeOutputsToSplit().replace_pattern(graph, match)
