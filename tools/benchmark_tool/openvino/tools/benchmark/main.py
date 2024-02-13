# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from datetime import datetime

from openvino.runtime import Dimension,properties

from openvino.tools.benchmark.benchmark import Benchmark
from openvino.tools.benchmark.parameters import parse_args
from openvino.tools.benchmark.utils.constants import MULTI_DEVICE_NAME, \
    CPU_DEVICE_NAME, GPU_DEVICE_NAME, \
    BLOB_EXTENSION, AUTO_DEVICE_NAME
from openvino.tools.benchmark.utils.inputs_filling import get_input_data
from openvino.tools.benchmark.utils.logging import logger
from openvino.tools.benchmark.utils.utils import next_step, get_number_iterations, pre_post_processing, \
    process_help_inference_string, print_perf_counters, print_perf_counters_sort, dump_exec_graph, get_duration_in_milliseconds, \
    get_command_line_arguments, parse_value_per_device, parse_devices, get_inputs_info, \
    print_inputs_and_outputs_info, get_network_batch_size, load_config, dump_config, get_latency_groups, \
    check_for_static, can_measure_as_static, parse_value_for_virtual_device, is_virtual_device, is_virtual_device_found
from openvino.tools.benchmark.utils.statistics_report import StatisticsReport, JsonStatisticsReport, CsvStatisticsReport, \
    averageCntReport, detailedCntReport
import openvino

def parse_and_check_command_line():
    def arg_not_empty(arg_value,empty_value):
        return not arg_value is None and not arg_value == empty_value

    parser = parse_args()
    args = parser.parse_args()

    if args.latency_percentile < 1 or args.latency_percentile > 100:
        parser.print_help()
        raise RuntimeError("The percentile value is incorrect. The applicable values range is [1, 100].")

    if not args.perf_hint == "none" and (arg_not_empty(args.number_streams, "") or arg_not_empty(args.number_threads, 0) or arg_not_empty(args.infer_threads_pinning, "")):
        raise Exception("-nstreams, -nthreads and -pin options are fine tune options. To use them you " \
                        "should explicitely set -hint option to none. This is not OpenVINO limitation " \
                        "(those options can be used in OpenVINO together), but a benchmark_app UI rule.")

    if args.report_type == "average_counters" and MULTI_DEVICE_NAME in args.target_device:
        raise Exception("only detailed_counters report type is supported for MULTI device")

    _, ext = os.path.splitext(args.path_to_model)
    is_network_compiled = True if ext == BLOB_EXTENSION else False
    is_precisiton_set = not (args.input_precision == "" and args.output_precision == "" and args.input_output_precision == "")

    if is_network_compiled and is_precisiton_set:
        raise Exception("Cannot set precision for a compiled model. " \
                        "Please re-compile your model with required precision.")

    return args, is_network_compiled

def main():
    args, is_network_compiled = parse_and_check_command_line()
    core = openvino.Core()
    core.set_property({"ENABLE_MMAP": False})
    core.read_model(args.model)
