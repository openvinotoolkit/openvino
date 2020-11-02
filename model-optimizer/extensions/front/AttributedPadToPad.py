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

from mo.front.common.replacement import FrontReplacementPattern
from mo.graph.graph import Graph, rename_nodes
from mo.ops.const import Const
from mo.ops.pad import Pad


class AttributedPadToPad(FrontReplacementPattern):
    """
    This transformation converts AttributedPad operation (begin/end paddings are specified as attribute) to Pad
    operation (Inference Engine semantic).
    """
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for attr_pad in graph.get_op_nodes(op='AttributedPad'):
            # save the original node name to use it in the new Pad op instance
            original_name = attr_pad.soft_get('name', attr_pad.id)

            new_pad = Pad(graph, {'mode': attr_pad.soft_get('mode', None), }).create_node()
            rename_nodes([(attr_pad, original_name + '/to_be_removed'), (new_pad, original_name)])

            attr_pad.in_port(0).get_connection().set_destination(new_pad.in_port(0))
            new_pad.in_port(1).connect(Const(graph, {'value': attr_pad.pads[:, 0]}).create_node().out_port(0))
            new_pad.in_port(2).connect(Const(graph, {'value': attr_pad.pads[:, 1]}).create_node().out_port(0))
            if attr_pad.soft_get('mode') == 'constant':
                new_pad.in_port(3).connect(Const(graph, {'value': attr_pad.fill_value}).create_node().out_port(0))

            attr_pad.out_port(0).get_connection().set_source(new_pad.out_port(0))
            graph.remove_node(attr_pad.id)
