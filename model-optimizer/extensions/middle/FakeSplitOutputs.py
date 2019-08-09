"""
 Copyright (c) 2019 Intel Corporation

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

    def pattern(self):
        return dict(
            nodes=[('split', dict(kind='op', op='SplitV'))],
            edges=[],
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        split = match['split']

        # Check that we need to add fake output (case when input.shape[axis] != sum(outputs.shape[axis])
        axis = split.axis
        input_shape = split.in_port(0).data.get_shape()[axis]

        output_shape = sum([split.out_node(port).shape[axis] for port in split.out_nodes()])

        # In such case we don't need to do anything
        if input_shape == output_shape:
            return

        # Adding fake outputs
        n_parts = int(input_shape/split.size_splits[0])
        part_shape = split.in_port(0).data.get_shape().copy()
        part_shape[axis] = split.size_splits[0]

        out_ports = split.out_ports()
        for i in range(n_parts):
            if i in out_ports and not split.out_port(i).disconnected():
                continue

            if i not in out_ports:
                split.add_output_port(i)

            output = Result(graph).create_node(attrs={'name': split.name + '/Fake_output_{}/'.format(i)})

            split.out_port(i).connect(output.in_port(0))
            output.in_port(0).data.set_shape(part_shape)
