# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from datetime import datetime

# from openvino.runtime import Dimension,properties

from openvino.tools.hyper_precision_checker.hyper_precision_checker import ModelRunner, ModelCreator
from openvino.tools.hyper_precision_checker.search_strategy import FP32FallbackSearcher
from openvino.tools.hyper_precision_checker.results_process import ResultChecker
from openvino.tools.hyper_precision_checker.inputs_filling import get_input_data
from openvino.tools.hyper_precision_checker.logging import logger
from openvino.tools.hyper_precision_checker.utils import parse_args, parser_perf_counters, get_inputs_info
from openvino.tools.hyper_precision_checker.constants import BIN_EXTENSION, XML_EXTENSION


def parse_and_check_command_line():
    def arg_not_empty(arg_value, empty_value):
        return not arg_value is None and not arg_value == empty_value
    args, parser = parse_args()
    return args


def init_model(path_to_model, path_to_model_bin, data_type, data_shape, iq_num=1):
    device = "CPU"
    # data_type = "fp32"
    config = {}
    batch_size = 1

    runner = ModelRunner(iq_num)
    runner.read_model(path_to_model, path_to_model_bin)
    runner.compile_model(data_type)
    requests = runner.create_infer_requests()
    logger.info(f'requests size: {len(requests)}')

    app_inputs_info = get_inputs_info(
        data_shape, batch_size, runner.model.inputs)

    return runner, app_inputs_info


def hyper_checker_infer(runner, data_queue):
    # use batch size according to provided layout and shapes
    latency_percentile = 0.9

    # Prepare input data
    if data_queue is None:
        logger.exception("data_queue must init first!")

    data_queue.reset()
    duration_ms = f"{runner.first_infer(data_queue):.2f}"
    logger.info(f"First inference took {duration_ms} ms")

    data_queue.reset()
    res_queue, fps, median_latency_ms, avg_latency_ms, min_latency_ms, max_latency_ms, total_duration_sec, iteration = runner.main_loop(
        data_queue, latency_percentile)

    perfs_count_list = []
    for request in runner.requests:
        perfs_count_list.append(request.profiling_info)
    return res_queue, perfs_count_list


def search_loop(path_to_model, path_to_model_bin, data_shape, data_queue, res_queue_fp32,
                threshold, results_cmp, searcher, new_model):
    # first round bf16 evaluation
    runner_bf16, _ = init_model(
        path_to_model, path_to_model_bin, "bf16", data_shape, 16)
    res_queue_bf16, perfs_count_list_bf16 = hyper_checker_infer(
        runner_bf16, data_queue)
    right, total = results_cmp.compare_results(res_queue_fp32, res_queue_bf16)
    accuracy_loss = 1.0 - right*1.0/total
    bf16_list, average_time = parser_perf_counters(perfs_count_list_bf16)

    searcher.init_searcher(bf16_list)
    searcher.store_config(right, average_time)

    if accuracy_loss <= threshold:
        logger.warning(
            f"BF16 Pass accuracy check!!! accuracy_loss={accuracy_loss:.3f} < {threshold}")
        return

    new_xml_path = "/tmp/tmp_model.xml"
    last_right = right

    while searcher.has_next():
        # generate potential fallback ops set
        potential_force_fp32_set = searcher.next_config()
        new_model.create_new_xml(new_xml_path, potential_force_fp32_set)

        runner, _ = init_model(
            new_xml_path, path_to_model_bin, "bf16", data_shape, 16)
        res_queue, perfs_count_list = hyper_checker_infer(runner, data_queue)

        right, total = results_cmp.compare_results(res_queue_fp32, res_queue)
        accuracy_loss = 1.0 - right*1.0/total

        if accuracy_loss < threshold:
            _, average_time = parser_perf_counters(perfs_count_list)
            searcher.store_config(right, average_time)
            logger.info(
                f"fallback {potential_force_fp32_set} to FP32 pass accuracy check!!! accuracy_loss={accuracy_loss:.3f} < {threshold}")
            break
        else:
            if last_right < right:
                logger.debug(f"### last_right={last_right}, right={right}")
                last_right = right
                _, average_time = parser_perf_counters(perfs_count_list)
                searcher.store_config(right, average_time)
            logger.info(
                f"accuracy_loss={accuracy_loss:.3f} > {threshold}, try to fallback more ops")
    return


def generate_input(paths_to_input, app_input_info, count):
    data_queue = get_input_data(paths_to_input, app_input_info, count)
    return data_queue


def main():
    try:
        # logger.info("Parsing input parameters")
        args = parse_and_check_command_line()

        model_filename = args.path_to_model
        data_shape = args.data_shape
        number_infer_requests = args.number_infer_requests
        threshold = args.threshold
        # if data_queue is NULL generate random data
        # run with fp32
        head, ext = os.path.splitext(model_filename)
        if ext == XML_EXTENSION:
            weights_filename = os.path.abspath(head + BIN_EXTENSION)
        else:
            raise Exception(
                "HyperChecker only support IR (xml) models")

        runner_fp32, app_inputs_info_fp32 = init_model(
            model_filename, weights_filename, "f32", data_shape, 16)
        data_queue = generate_input(
            None, app_inputs_info_fp32, number_infer_requests)
        res_queue_fp32, perfs_count_list_fp32 = hyper_checker_infer(
            runner_fp32, data_queue)
        _, average_time = parser_perf_counters(perfs_count_list_fp32)

        searcher = FP32FallbackSearcher(number_infer_requests, average_time)
        new_model = ModelCreator(model_filename)
        result_cmp = ResultChecker()
        result_cmp.set_outputs([0])

        search_loop(model_filename, weights_filename, data_shape, data_queue,
                    res_queue_fp32, threshold, result_cmp, searcher, new_model)

        best_set = None
        best_accucary = 0
        best_latency = 0.0
        for it in searcher.best_config:
            if it[0] != "all" and it[1] > best_accucary:
                best_accucary = it[1]
                best_latency = it[2]
                best_set = it[0]
            logger.info(
                f"accuracy={it[1]/number_infer_requests:.3f}, force_use_fp32_ops={it[0]}, average_latency={it[2]:.3f}")
        logger.info(
            f"best config = {best_set}, average_latency={best_latency:.3f}, accuracy={best_accucary/number_infer_requests:.3f}")

        if best_set == "None":
            logger.info("All ops can run under bf16!")
        else:
            new_path = model_filename + '.new'
            logger.info(
                f'generate new xml {new_path}, with {best_set} fallback to FP32, '
                'accuracy={best_accucary/number_infer_requests:.3f}')
            new_model.create_new_xml(new_path, best_set)

    except Exception as e:
        logger.exception(e)
        sys.exit(1)
