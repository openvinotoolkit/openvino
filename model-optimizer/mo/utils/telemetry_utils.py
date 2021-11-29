# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
from collections import Counter

import numpy as np

from mo.graph.graph import Graph
from mo.middle.pattern_match import for_graph_and_each_sub_graph_recursively
from mo.utils.cli_parser import get_params_with_paths_list

try:
    import openvino_telemetry as tm
except ImportError:
    import mo.utils.telemetry_stub as tm


def send_op_names_info(framework: str, graph: Graph):
    """
    This function sends information about operations in model.
    :param framework: framework name.
    :param graph: model graph.
    """
    op_counter = Counter()

    def gather_op_statistics(g: Graph, op_c: Counter = op_counter):
        if hasattr(g, 'op_names_statistic'):
            op_c += g.op_names_statistic

    for_graph_and_each_sub_graph_recursively(graph, gather_op_statistics)

    t = tm.Telemetry()
    for op_name in op_counter:
        t.send_event('mo', 'op_count', "{}_{}".format(framework, op_name), op_counter[op_name])


def send_shapes_info(framework: str, graph: Graph):
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
            shape_str += np.array2string(shape) + ","
            if not all(shape > 0):
                is_partially_defined = "1"
        message_str = "{fw:" + framework + ",shape:\"" + shape_str[:-1] + "\"}"
        t.send_event('mo', 'input_shapes', message_str)
        t.send_event('mo', 'partially_defined_shape',
                     "{partially_defined_shape:" + is_partially_defined + ",fw:" + framework + "}")


def send_params_info(argv: argparse.Namespace, cli_parser: argparse.ArgumentParser):
    """
    This function sends information about used command line parameters.
    :param argv: command line parameters.
    :param cli_parser: command line parameters parser.
    """
    t = tm.Telemetry()
    params_with_paths = get_params_with_paths_list()
    for arg in vars(argv):
        arg_value = getattr(argv, arg)
        if arg_value != cli_parser.get_default(arg):
            if arg in params_with_paths:
                # If command line argument value is a directory or a path to file it is not sent
                # as it may contain confidential information. "1" value is used instead.
                param_str = arg + ":" + str(1)
            else:
                param_str = arg + ":" + str(arg_value)

            t.send_event('mo', 'cli_parameters', param_str)


def send_framework_info(framework: str):
    """
    This function sends information about used framework.
    :param framework: framework name.
    """
    t = tm.Telemetry()
    t.send_event('mo', 'framework', framework)
