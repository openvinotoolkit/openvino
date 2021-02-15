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

from extensions.ops.elementwise import Add
from mo.back.replacement import BackReplacementPattern
from mo.front.common.partial_infer.utils import int64_array
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph, rename_nodes, rename_node


class FakeOutputResolver(BackReplacementPattern):
    """
    This transformation removes FakeOutput nodes. If producer of FakeOutput have only one consumer (FakeOutput itself)
     the name of FakeOutput is inherited by its producer, otherwise FakeOutput is replaced with op which does nothing.
    """
    enabled = True
    force_clean_up = True

    def find_and_replace_pattern(self, graph: Graph):
        for fake_output in graph.get_op_nodes(op='FakeOutput'):
            name = fake_output.soft_get('name', fake_output.id)

            producer = fake_output.in_port(0).get_source().node
            producer_outputs = 0
            for port in producer.out_ports().values():
                if not port.disconnected():
                    producer_outputs += 1
            if producer_outputs != 1:
                # At this stage we don't know the type of output, so we rely on MO transformation which updates the
                # Const type for elementwise operations in case of input data types mismatch
                add = create_op_with_const_inputs(graph, Add, {1: int64_array(0)}, {'can_be_fused': False})
                rename_nodes([(fake_output, name + '/TBD'), (add, name)])

                fake_output.in_port(0).get_connection().set_destination(add.in_port(0))
                fake_output.out_port(0).get_connection().set_source(add.out_port(0))
            else:
                result_in_port = fake_output.out_port(0).get_destination()
                result_in_port.disconnect()
                fake_output.in_port(0).get_connection().set_destination(result_in_port)
                rename_nodes([(fake_output, name + '/TBD'), (producer, name)])
