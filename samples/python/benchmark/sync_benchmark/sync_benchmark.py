#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
import statistics
import sys
from time import perf_counter

import numpy as np
import openvino as ov
from openvino.utils.types import get_dtype


def fill_tensor_random(tensor):
    dtype = get_dtype(tensor.element_type)
    rand_min, rand_max = (0, 1) if dtype == bool else (np.iinfo(np.uint8).min, np.iinfo(np.uint8).max)
    # np.random.uniform excludes high: add 1 to have it generated
    if np.dtype(dtype).kind in ['i', 'u', 'b']:
        rand_max += 1
    rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(0)))
    if 0 == tensor.get_size():
        raise RuntimeError("Models with dynamic shapes aren't supported. Input tensors must have specific shapes before inference")
    tensor.data[:] = rs.uniform(rand_min, rand_max, list(tensor.shape)).astype(dtype)


def main():
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)
    log.info('OpenVINO:')
    log.info(f"{'Build ':.<39} {ov.__version__}")
    device_name = 'CPU'
    if len(sys.argv) == 3:
        device_name = sys.argv[2]
    elif len(sys.argv) != 2:
        log.info(f'Usage: {sys.argv[0]} <path_to_model> <device_name>(default: CPU)')
        return 1
    # Optimize for latency. Most of the devices are configured for latency by default,
    # but there are exceptions like GNA
    latency = {'PERFORMANCE_HINT': 'LATENCY'}

    # Create Core and use it to compile a model.
    # Select the device by providing the name as the second parameter to CLI.
    # Using MULTI device is pointless in sync scenario
    # because only one instance of openvino.runtime.InferRequest is used
    core = ov.Core()
    compiled_model = core.compile_model(sys.argv[1], device_name, latency)
    ireq = compiled_model.create_infer_request()
    # Fill input data for the ireq
    for model_input in compiled_model.inputs:
        fill_tensor_random(ireq.get_tensor(model_input))
    # Warm up
    ireq.infer()
    # Benchmark for seconds_to_run seconds and at least niter iterations
    seconds_to_run = 10
    niter = 10
    latencies = []
    start = perf_counter()
    time_point = start
    time_point_to_finish = start + seconds_to_run
    while time_point < time_point_to_finish or len(latencies) < niter:
        ireq.infer()
        iter_end = perf_counter()
        latencies.append((iter_end - time_point) * 1e3)
        time_point = iter_end
    end = time_point
    duration = end - start
    # Report results
    fps = len(latencies) / duration
    log.info(f'Count:          {len(latencies)} iterations')
    log.info(f'Duration:       {duration * 1e3:.2f} ms')
    log.info('Latency:')
    log.info(f'    Median:     {statistics.median(latencies):.2f} ms')
    log.info(f'    Average:    {sum(latencies) / len(latencies):.2f} ms')
    log.info(f'    Min:        {min(latencies):.2f} ms')
    log.info(f'    Max:        {max(latencies):.2f} ms')
    log.info(f'Throughput: {fps:.2f} FPS')


if __name__ == '__main__':
    main()
