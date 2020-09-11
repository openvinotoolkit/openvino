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


class RemoveUselessConvert(BackReplacementPattern):
    """
    Transformation looks for the Converts layers that do not change actual tensor data type.
    The transformation is executed explicitly from the prepare_emit_ir function
    """
    enabled = False
    run_not_recursively = True

    def find_and_replace_pattern(self, graph: Graph):
        for cast_node in graph.get_op_nodes(op='Cast'):
            if cast_node.in_port(0).get_data_type() == cast_node.out_port(0).get_data_type():
                log.debug('Convert node {} do not change the data type of the input data.'.format(cast_node.name))
                cast_node.out_port(0).get_connection().set_source(cast_node.in_port(0).get_connection().get_source())
                graph.remove_node(cast_node.id)
