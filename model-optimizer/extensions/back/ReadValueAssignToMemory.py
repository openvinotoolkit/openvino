"""
 Copyright (C) 2020 Intel Corporation

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
from extensions.back.CutMemory import CutMemoryInput, CutMemoryOutput
from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph
from mo.ops.memory import Memory


"""
 All transformations in this file should be removed after removing IR v7 support
"""


class ReplaceReadValueByMemory(BackReplacementPattern):
    """
    Replace ReadValue by Memory. Should be removed after v7 IR support removing.
    """
    enabled = True
    graph_condition = [lambda graph: not graph.graph['cmd_params'].generate_experimental_IR_V10]
    force_clean_up = True

    def run_after(self):
        return [CutMemoryInput]

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('op', dict(kind='op', op='ReadValue'))],
            edges=[]
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        node = match['op']
        node_id = node['variable_id']

        node.in_port(0).disconnect()
        new_in = Memory(graph, {'name': node.id, 'id': node_id, 'index': 1, 'size': 2,
                                'shape': list(node.out_port(0).data.get_shape())[1:]}).create_node()
        for dest in node.out_port(0).get_destinations():
            dest.disconnect()
            new_in.out_port(0).connect(dest)


class ReplaceAssignByMemory(BackReplacementPattern):
    """
    Replace Assign by Memory. Should be removed after v7 IR support removing.
    """
    enabled = True
    graph_condition = [lambda graph: not graph.graph['cmd_params'].generate_experimental_IR_V10]
    force_clean_up = True

    def run_after(self):
        return [CutMemoryOutput]

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('op', dict(kind='op', op='Assign'))],
            edges=[]
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        node = match['op']
        node_id = node['variable_id']

        new_out = Memory(graph, {'name': node.id, 'id': node_id, 'index': 0, 'size': 2,
                                 'shape': list(node.out_port(0).data.get_shape())[1:]}).create_node()
        node.in_port(0).get_source().connect(new_out.in_port(0))
        node.in_port(0).disconnect()
        node.out_port(0).get_connection().set_source(new_out.out_port(0))


class KaldiRemoveMemoryOutputBackReplacementPatternV7(BackReplacementPattern):
    enabled = True
    graph_condition = [lambda graph: not graph.graph['cmd_params'].generate_experimental_IR_V10]

    def run_after(self):
        from extensions.back.pass_separator import BackFinish
        return [BackFinish]

    def run_before(self):
        from extensions.back.SpecialNodesFinalization import RemoveOutputOps
        return [RemoveOutputOps]

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('memory_node', dict(op='Memory')),
                ('data_node', dict(kind='data')),
                ('op_output', dict(op='Result'))
            ],
            edges=[
                ('memory_node', 'data_node'),
                ('data_node', 'op_output')
            ]
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        """
        Need to find the pattern: Memory -> Data -> Result

        It is needed to make Memory nodes appear in IR,
        but they are output nodes by default and we remove the Result node after each output memory.

        DO NOT use graph clean up after it
        otherwise Memory nodes would be removed as they are not on the path from input to output

        Parameters
        ----------
        graph : Graph
           Graph with loaded model.
        match : dict
           Patterns which were found in graph structure.
        """
        memory = match['memory_node']
        data = match['data_node']
        op_output = match['op_output']

        graph.remove_edge(memory.id, data.id)
        graph.remove_node(data.id)
        graph.remove_node(op_output.id)
