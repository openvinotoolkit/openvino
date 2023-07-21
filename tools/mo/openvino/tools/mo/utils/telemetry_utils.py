# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import argparse
import numbers
from collections import Counter

import numpy as np
from openvino.runtime import get_version as get_rt_version  # pylint: disable=no-name-in-module,import-error

from openvino.tools.mo.front.common.partial_infer.utils import is_fully_defined, unmask_shape, int64_array
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.middle.pattern_match import for_graph_and_each_sub_graph_recursively
from openvino.tools.mo.utils.cli_parser import get_params_with_paths_list
from openvino.tools.mo.utils.telemetry_params import telemetry_params
from openvino.tools.mo.utils.utils import check_values_equal

try:
    import openvino_telemetry as tm
    from openvino_telemetry.backend import backend_ga4
except ImportError:
    import openvino.tools.mo.utils.telemetry_stub as tm


def init_mo_telemetry(app_name='Model Optimizer'):
    return tm.Telemetry(tid=get_tid(), app_name=app_name, app_version=get_rt_version(), backend='ga4')


def send_framework_info(framework: str):
    """
    This function sends information about used framework.
    :param framework: framework name.
    """
    t = tm.Telemetry()
    t.send_event('mo', 'framework', framework)


def get_tid():
    """
    This function returns the ID of the database to send telemetry.
    """
    return telemetry_params['TID']


def send_conversion_result(conversion_result: str, need_shutdown=False):
    t = tm.Telemetry()
    t.send_event('mo', 'conversion_result', conversion_result)
    t.end_session('mo')
    if need_shutdown:
        t.force_shutdown(1.0)


def arg_to_str(arg):
    # This method converts to string only known types, otherwise returns string with name of the type
    from openvino.runtime import PartialShape, Shape, Type, Layout # pylint: disable=no-name-in-module,import-error
    if isinstance(arg, (PartialShape, Shape, Type, Layout)):
        return str(arg)
    if isinstance(arg, (str, numbers.Number, bool)):
        return str(arg)
    return str(type(arg))


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
        if not check_values_equal(arg_value, cli_parser.get_default(arg)):
            if arg in params_with_paths:
                # If command line argument value is a directory or a path to file it is not sent
                # as it may contain confidential information. "1" value is used instead.
                param_str = arg + ":" + str(1)
            else:
                param_str = arg + ":" + arg_to_str(arg_value)

            t.send_event('mo', 'cli_parameters', param_str)


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
            shape_str += (np.array2string(int64_array(unmask_shape(shape))) if shape is not None else "Undefined") + ","
            if not is_fully_defined(shape):
                is_partially_defined = "1"
        message_str = "{fw:" + framework + ",shape:\"" + shape_str[:-1] + "\"}"
        t.send_event('mo', 'input_shapes', message_str)
        t.send_event('mo', 'partially_defined_shape',
                     "{partially_defined_shape:" + is_partially_defined + ",fw:" + framework + "}")
