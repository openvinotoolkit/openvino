#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
from math import ceil
import sys
from time import perf_counter

import numpy as np
from openvino.runtime import Core, get_version, AsyncInferQueue
from openvino.runtime.utils.types import get_dtype


def percentile(values, percent):
    return values[ceil(len(values) * percent / 100) - 1]


def fill_tensor_random(tensor):
    dtype = get_dtype(tensor.element_type)
    rand_min, rand_max = (0, 1) if dtype == np.bool else (np.iinfo(np.uint8).min, np.iinfo(np.uint8).max)
    # np.random.uniform excludes high: add 1 to have it generated
    if np.dtype(dtype).kind in ['i', 'u', 'b']:
        rand_max += 1
    rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(0)))
    if 0 == tensor.get_size():
        raise RuntimeError("Models with dynamic shapes aren't supported. Input tensors must have specific shapes before inference");
    tensor.data[:] = rs.uniform(rand_min, rand_max, list(tensor.shape)).astype(dtype)


# TODO: make file executable
def main():
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)
    log.info(f"OpenVINO:\n{'': <9}{'API version':.<24} {get_version()}")
    # Parsing and validation of input arguments
    if len(sys.argv) != 2:
        log.info(f'Usage: {sys.argv[0]} <path_to_model>')
        return 1
    core = Core()

    tput = {"PERFORMANCE_HINT": "THROUGHPUT"}
    # TODO: try MULTI:CPU,BATCH:GPU(4)
    compiled_model = core.compile_model(sys.argv[1], "CPU", tput)
    ireqs = AsyncInferQueue(compiled_model)
    nireq = len(ireqs)
    for ireq in ireqs:
        for input in compiled_model.inputs:
            fill_tensor_random(ireq.get_tensor(input))

    init_niter = 1000
    # Align number of iterations by request number
    niter = int((init_niter + nireq - 1) / nireq) * nireq
    if init_niter != niter:
        log.warning('Number of iterations was aligned by request number '
                    f'from {init_niter} to {niter} using number of requests {nireq}')

    times = []
    in_fly = set()
    start_time = perf_counter()
    for _ in range(niter):
        idle_id = ireqs.get_idle_request_id()
        if idle_id in in_fly:
            times.append(ireqs[idle_id].latency)
        else:
            in_fly.add(idle_id)
        ireqs.start_async()

    ireqs.wait_all()
    duration = perf_counter() - start_time
    for infer_request_id in in_fly:
        times.append(ireqs[infer_request_id].latency)

    times.sort()
    percent = 50
    percentile_latency_ms = percentile(times, percent)
    avg_latency_ms = sum(times) / len(times)
    min_latency_ms = times[0]
    max_latency_ms = times[-1]
    fps = niter / duration
    print(f'Count:          {niter} iterations')
    print(f'Duration:       {duration * 1e3:.2f} ms')
    print('Latency:')
    if percent == 50:
        print(f'    Median:     {percentile_latency_ms:.2f} ms')
    else:
        print(f'({percent} percentile):     {percentile_latency_ms:.2f} ms')
    print(f'    AVG:        {avg_latency_ms:.2f} ms')
    print(f'    MIN:        {min_latency_ms:.2f} ms')
    print(f'    MAX:        {max_latency_ms:.2f} ms')
    print(f'Throughput: {fps:.2f} FPS')


if __name__ == "__main__":
    main()
