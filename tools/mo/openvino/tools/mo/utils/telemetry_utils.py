# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from collections import Counter
import numpy as np
from openvino.tools.mo.middle.pattern_match import for_graph_and_each_sub_graph_recursively
from openvino.tools.mo.front.common.partial_infer.utils import is_fully_defined, unmask_shape, int64_array
from openvino.tools.mo_lite.utils.telemetry_utils import send_framework_info
try:
    import openvino_telemetry as tm
except ImportError:
    import openvino.tools.mo_lite.utils.telemetry_stub as tm


def send_op_names_info(framework: str, graph):
    """
    This function sends information about operations in model.
    :param framework: framework name.
    :param graph: model graph.
    """
    op_counter = Counter()

    def gather_op_statistics(g, op_c: Counter = op_counter):
        if hasattr(g, 'op_names_statistic'):
            op_c += g.op_names_statistic

    for_graph_and_each_sub_graph_recursively(graph, gather_op_statistics)

    t = tm.Telemetry()
    for op_name in op_counter:
        t.send_event('mo', 'op_count', "{}_{}".format(framework, op_name), op_counter[op_name])


def send_shapes_info(framework: str, graph):
    """
    This function sends information about model input shapes.
    :param framework: framework name.
    :param graph: model graph.
    """
    shapes = []
    for node in graph.get_op_nodes():
        op_type = node.soft_get('type', None)
        if op_type == 'Parameter':
            if 'shape' in node:
                shapes.append(node['shape'])
    t = tm.Telemetry()

    if shapes:
        shape_str = ""
        is_partially_defined = "0"
        for shape in shapes:
            shape_str += (np.array2string(int64_array(unmask_shape(shape))) if shape is not None else "Undefined") + ","
            if not is_fully_defined(shape):
                is_partially_defined = "1"
        message_str = "{fw:" + framework + ",shape:\"" + shape_str[:-1] + "\"}"
        t.send_event('mo', 'input_shapes', message_str)
        t.send_event('mo', 'partially_defined_shape',
                     "{partially_defined_shape:" + is_partially_defined + ",fw:" + framework + "}")
