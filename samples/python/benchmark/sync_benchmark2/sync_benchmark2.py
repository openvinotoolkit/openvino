#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import datetime
import logging as log
from math import ceil
import sys
from time import perf_counter

import numpy as np
from openvino.runtime import Core, get_version
from openvino.runtime.utils.types import get_dtype


def percentile(values, percent):
    return values[ceil(len(values) * percent / 100) - 1]


def get_random_inputs(compiled_model):
    random_inputs = {}
    for model_input in compiled_model.inputs:
        tensor_descr = model_input.tensor
        dtype = get_dtype(tensor_descr.element_type)
        rand_min, rand_max = (0, 1) if dtype == np.bool else (np.iinfo(np.uint8).min, np.iinfo(np.uint8).max)
        # np.random.uniform excludes high: add 1 to have it generated
        if np.dtype(dtype).kind in ['i', 'u', 'b']:
            rand_max += 1
        rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(0)))
        if 0 == tensor_descr.size:
            raise RuntimeError("Models with dynamic shapes aren't supported. Input tensors must have specific shapes before inference")
        random_inputs[model_input.any_name] = rs.uniform(rand_min, rand_max, list(tensor_descr.shape)).astype(dtype)
    return random_inputs


def print_perf_counters(perf_counts_list):
    max_layer_name = 30
    for ni in range(len(perf_counts_list)):
        perf_counts = perf_counts_list[ni]
        total_time = datetime.timedelta()
        total_time_cpu = datetime.timedelta()
        log.info(f"Performance counts for {ni}-th infer request")
        for pi in perf_counts:
            print(f"{pi.node_name[:max_layer_name - 4] + '...' if (len(pi.node_name) >= max_layer_name) else pi.node_name:<30}"
                                                                f"{str(pi.status):<15}"
                                                                f"{'layerType: ' + pi.node_type:<30}"
                                                                f"{'realTime: ' + str(pi.real_time):<20}"
                                                                f"{'cpu: ' +  str(pi.cpu_time):<20}"
                                                                f"{'execType: ' + pi.exec_type:<20}")
            total_time += pi.real_time
            total_time_cpu += pi.cpu_time
        log.info(f'Total time:     {total_time} seconds')
        log.info(f'Total CPU time: {total_time_cpu} seconds\n')


def main():
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)
    log.info(f"OpenVINO:\n{'': <9}{'API version':.<24} {get_version()}")
    if len(sys.argv) != 2:
        log.info(f'Usage: {sys.argv[0]} <path_to_model>')
        return 1
    # Optimize for latency. Most of the devices are configured for latency by default,
    # but there are exceptions like MYRIAD
    latency = {'PERFORMANCE_HINT': 'LATENCY'}

    # Create Core and use it to compile a model
    # Pick device by replacing CPU, for example AUTO:GPU,CPU.
    # Using MULTI device is pointless in sync scenario
    # because only one instance of openvino.runtime.InferRequest is used
    core = Core()
    compiled_model = core.compile_model(sys.argv[1], 'CPU', latency)
    random_inputs = get_random_inputs(compiled_model)
    # Warm up
    compiled_model.infer_new_request(random_inputs)
    # Benchmark for seconds_to_run seconds and at least niter iterations
    seconds_to_run = 15
    niter = 10
    latencies = []
    start = perf_counter()
    time_point = start
    time_point_to_finish = start + seconds_to_run
    while time_point < time_point_to_finish or len(latencies) < niter:
        compiled_model.infer_new_request(random_inputs)
        iter_end = perf_counter()
        latencies.append((iter_end - time_point) * 1e3)
        time_point = iter_end
    end = time_point
    duration = end - start
    # Report results

    latencies.sort()
    percent = 50
    percentile_latency_ms = percentile(latencies, percent)
    avg_latency_ms = sum(latencies) / len(latencies)
    min_latency_ms = latencies[0]
    max_latency_ms = latencies[-1]
    fps = len(latencies) / duration
    log.info(f'Count:          {len(latencies)} iterations')
    log.info(f'Duration:       {duration * 1e3:.2f} ms')
    log.info('Latency:')
    if percent == 50:
        log.info(f'    Median:     {percentile_latency_ms:.2f} ms')
    else:
        log.info(f'({percent} percentile):     {percentile_latency_ms:.2f} ms')
    log.info(f'    AVG:        {avg_latency_ms:.2f} ms')
    log.info(f'    MIN:        {min_latency_ms:.2f} ms')
    log.info(f'    MAX:        {max_latency_ms:.2f} ms')
    log.info(f'Throughput: {fps:.2f} FPS')


if __name__ == '__main__':
    main()
