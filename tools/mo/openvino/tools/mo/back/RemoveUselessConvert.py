# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

from openvino.tools.mo.back.replacement import BackReplacementPattern
from openvino.tools.mo.graph.graph import Graph


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
