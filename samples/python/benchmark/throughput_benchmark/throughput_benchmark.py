#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging as log
import sys
import statistics
import signal
import threading
from time import perf_counter

import numpy as np
import openvino as ov
from openvino.utils.types import get_dtype

_stop_event = threading.Event()


def signal_handler(sig, frame):
    _stop_event.set()


def register_signal_handlers():
    for name in ('SIGINT', 'SIGTERM', 'SIGBREAK'):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, signal_handler)
        except (ValueError, OSError, RuntimeError):
            # It may throw if not supported not main thread etc.
            pass


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
    parser = argparse.ArgumentParser()
    parser.add_argument('path_to_model', help='Path to model')
    parser.add_argument('device_name', nargs='?', default='CPU')
    parser.add_argument('--seconds-to-run', type=int, default=10, help='Duration in seconds; 0 runs until a stop signal')
    parser.add_argument('--niter', type=int, default=10, help='Minimum number of iterations')
    args = parser.parse_args()
    if args.seconds_to_run < 0 or args.niter < 1:
        parser.error('--seconds-to-run must be nonnegative and --niter must be positive')

    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)
    log.info('OpenVINO:')
    log.info(f"{'Build ':.<39} {ov.__version__}")
    # Optimize for throughput. Best throughput can be reached by
    # running multiple openvino.InferRequest instances asynchronously
    tput = {'PERFORMANCE_HINT': 'THROUGHPUT'}

    # Create Core and use it to compile a model.
    # Select the device by providing the name as the second parameter to CLI.
    # It is possible to set CUMULATIVE_THROUGHPUT as PERFORMANCE_HINT for AUTO device
    core = ov.Core()
    compiled_model = core.compile_model(args.path_to_model, args.device_name, tput)
    # AsyncInferQueue creates optimal number of InferRequest instances
    ireqs = ov.AsyncInferQueue(compiled_model)
    # Fill input data for ireqs
    for ireq in ireqs:
        for model_input in compiled_model.inputs:
            fill_tensor_random(ireq.get_tensor(model_input))

    # Warm up
    for _ in range(len(ireqs)):
        ireqs.start_async()
    ireqs.wait_all()

    # Benchmark for seconds_to_run seconds and at least niter iterations
    seconds_to_run = args.seconds_to_run
    niter = args.niter
    latencies = []
    in_fly = set()

    register_signal_handlers()

    start = perf_counter()
    time_point_to_finish = start + seconds_to_run

    log.info('Starting inference loop')
    while seconds_to_run == 0 or perf_counter() < time_point_to_finish or len(latencies) + len(in_fly) < niter:
        if _stop_event.is_set():
            log.info('Stop signal received, finishing in-flight requests and reporting stats...')
            break

        idle_id = ireqs.get_idle_request_id()

        if idle_id in in_fly:
            latencies.append(ireqs[idle_id].latency)
        else:
            in_fly.add(idle_id)

        ireqs.start_async()

    ireqs.wait_all()
    duration = perf_counter() - start
    for infer_request_id in in_fly:
        latencies.append(ireqs[infer_request_id].latency)

    if not latencies:
        log.info('No completed iterations to report.')
        return 0

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

    return 0


if __name__ == '__main__':
    main()
