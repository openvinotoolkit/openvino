# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from datetime import datetime

# from openvino.runtime import Dimension,properties

from openvino.tools.hyper_precision_checker.hyper_precision_checker import ModelRunner, ModelCreator
from openvino.tools.hyper_precision_checker.inputs_filling import get_input_data
from openvino.tools.hyper_precision_checker.logging import logger
from openvino.tools.hyper_precision_checker.utils import parse_args, generate_bf16_ops_list, parser_perf_counters, get_inputs_info
from openvino.tools.hyper_precision_checker.constants import BIN_EXTENSION, XML_EXTENSION
from openvino.tools.hyper_precision_checker.results_process import OV_Result, ResultChecker

def parse_and_check_command_line():
    def arg_not_empty(arg_value,empty_value):
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
 

def search_loop(path_to_model, path_to_model_bin, data_shape, data_queue, res_queue_fp32, threshold, results_cmp):
    force_use_fp32_ops = set()
    best_config = []
    # first round bf16 evaluation
    runner_bf16, _ = init_model(path_to_model, path_to_model_bin, "bf16", data_shape, 16)
    
    res_queue_bf16, perfs_count_list_bf16 = hyper_checker_infer(runner_bf16, data_queue)

    right, total = results_cmp.compare_results(res_queue_fp32, res_queue_bf16)

    accuracy_loss = 1.0 - right*1.0/total
    logger.info(
        f"accuracy loss between FP32 and BF16 is {accuracy_loss}")

    bf16_list, average_time = parser_perf_counters(perfs_count_list_bf16)

    best_config.append([force_use_fp32_ops.copy(), right, average_time])
    if accuracy_loss <= threshold:
        print(
            f"BF16 Pass accuracy check!!! accuracy_loss={accuracy_loss} < {threshold}")
        return best_config

    potential_bf16_ops_list = generate_bf16_ops_list(bf16_list)
        
    new_model = ModelCreator(path_to_model)
    new_xml_path = "/tmp/tmp_model.xml"
    last_right = right
    
    while len(potential_bf16_ops_list) > 0:
        print(
            f"%%%  best_config={best_config}   {len(potential_bf16_ops_list)} round to search %%%")
        potential_force_fp32_set = force_use_fp32_ops.copy()
        try_op = potential_bf16_ops_list.pop(0)
        logger.info(f"try to fallback {try_op} to FP32")
        potential_force_fp32_set.add(try_op)
        logger.info(
            f"{potential_force_fp32_set} will fallback to FP32")

        new_model.create_new_xml(new_xml_path, potential_force_fp32_set)

        runner, app_inputs_info = init_model(new_xml_path, path_to_model_bin, "bf16", data_shape, 16)
        res_queue = None
        res_queue, perfs_count_list = hyper_checker_infer(runner, data_queue)

        right, total = results_cmp.compare_results(res_queue_fp32, res_queue)
        accuracy_loss = 1.0 - right*1.0/total

        if accuracy_loss < threshold:
            force_use_fp32_ops.add(try_op)
            print(
                f"fallback {force_use_fp32_ops} to FP32 pass accuracy check!!! accuracy_loss={accuracy_loss} < {threshold}")
            break
        else:
            if last_right < right:
                print(f"### last_right={last_right}, right={right}")
                last_right = right
                force_use_fp32_ops.add(try_op)
                _, average_time = parser_perf_counters(perfs_count_list)
                best_config.append(
                    [force_use_fp32_ops.copy(), last_right, average_time])
            print(
                f"accuracy_loss={accuracy_loss} > {threshold}, try to fallback more ops")
    return best_config
    

def generate_input(paths_to_input, app_input_info, count):
    data_queue = get_input_data(paths_to_input, app_input_info, count)
    return data_queue


def main():
    try:
        # logger.info("Parsing input parameters")
        args = parse_and_check_command_line()
       
        # if data_queue is NULL generate random data
        #run with fp32
        model_filename = args.path_to_model
        head, ext = os.path.splitext(model_filename)
        weights_filename = os.path.abspath(head + BIN_EXTENSION) if ext == XML_EXTENSION else None
        runner_fp32, app_inputs_info_fp32 = init_model(
            model_filename, weights_filename, "f32", args.data_shape, 16)
        data_queue = generate_input(None, app_inputs_info_fp32, args.number_infer_requests)
        res_queue_fp32, perfs_count_list_fp32 = hyper_checker_infer(runner_fp32, data_queue)
        
        _, average_time = parser_perf_counters(perfs_count_list_fp32)


        result_cmp = ResultChecker()
        result_cmp.set_outputs([0])
        
        best_config = search_loop(model_filename, weights_filename, args.data_shape,
                                  data_queue, res_queue_fp32, args.threshold, result_cmp)
        best_config.append(['all', 1.0, average_time])
        for it in best_config:
            print(
                f"accuracy={it[1]/args.number_infer_requests}, force_use_fp32_ops={it[0]}, average_latency={it[2]}")
            
    except Exception as e:
        logger.exception(e)
        sys.exit(1)
       