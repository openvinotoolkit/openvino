#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import argparse
import logging as log
import sys
from timeit import default_timer

import cv2
import numpy as np
from openvino.inference_engine import ExecutableNetwork, IECore


def parse_args() -> argparse.Namespace:
    '''Parse and return command line arguments'''
    parser = argparse.ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', help='Show this help message and exit.')
    args.add_argument('-m', '--model', required=True, type=str,
                      help='Required. Path to an .xml or .onnx file with a trained model.')
    args.add_argument('-i', '--input', required=True, type=str, help='Required. Path to an image file.')
    args.add_argument('-r', '--reference', type=str,
                      help='Optional. Read reference score .ark file and compare scores.')
    args.add_argument('-d', '--device', default='CPU', type=str,
                      help='Optional. Specify the target device to infer on; CPU, GPU, MYRIAD, HDDL or HETERO: '
                      'is acceptable. The sample will look for a suitable plugin for device specified. '
                      'Default value is CPU.')
    args.add_argument('-bs', '--batch_size', default=1, type=int, help='Optional. Default 1')

    return parser.parse_args()


def read_ark_file(file_name: str) -> list:
    '''Read utterance matrices from a .ark file'''
    def read_key(file) -> str:
        '''Read a identifier of utterance matrix'''
        key = ''
        while True:
            char = file.read(1).decode()
            if char in ('', ' '):
                break
            else:
                key += char

        return key

    def read_matrix(file) -> np.ndarray:
        '''Read a utterance matrix'''
        header = file.read(5).decode()
        if 'FM' in header:
            num_of_bytes = 4
            dtype = 'float32'
        elif 'DM' in header:
            num_of_bytes = 8
            dtype = 'float64'

        _, rows, _, cols = np.frombuffer(file.read(10), 'int8, int32, int8, int32')[0]
        buffer = file.read(rows * cols * num_of_bytes)
        vector = np.frombuffer(buffer, dtype)
        matrix = np.reshape(vector, (rows, cols))

        return matrix

    matrix_list = []
    with open(file_name, 'rb') as file:
        while read_key(file):
            matrix_list.append(read_matrix(file))

    return matrix_list


def infer_matrix(matrix: np.ndarray, exec_net: ExecutableNetwork, input_blob: str, out_blob: str) -> np.ndarray:
    '''Do a synchronous matrix inference'''
    batch_size, num_of_dims = exec_net.outputs[out_blob].shape
    result = np.ndarray((matrix.shape[0], num_of_dims))

    slice_begin = 0
    slice_end = batch_size

    while True:
        vectors = matrix[slice_begin:slice_end]

        if vectors.shape[0] < batch_size:
            for vector in vectors:
                vector_result = exec_net.infer({input_blob: vector})
                result[slice_begin] = vector_result[out_blob][0]
                slice_begin += 1
        else:
            vector_results = exec_net.infer({input_blob: vectors})
            result[slice_begin:slice_end] = vector_results[out_blob]
            slice_begin += batch_size
            slice_end += batch_size

        if slice_begin >= matrix.shape[0]:
            return result


def main():
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)
    args = parse_args()

# ---------------------------Step 1. Initialize inference engine core--------------------------------------------------
    log.info('Creating Inference Engine')
    ie = IECore()

# ---------------------------Step 2. Read a model in OpenVINO Intermediate Representation or ONNX format---------------
    log.info(f'Reading the network: {args.model}')
    # (.xml and .bin files) or (.onnx file)
    net = ie.read_network(model=args.model)

# ---------------------------Step 3. Configure input & output----------------------------------------------------------
    log.info('Configuring input and output blobs')
    # Get names of input and output blobs
    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))

    # Set input and output precision manually
    net.input_info[input_blob].precision = 'FP32'
    net.outputs[out_blob].precision = 'FP32'
    net.batch_size = args.batch_size

# ---------------------------Step 4. Loading model to the device-------------------------------------------------------
    log.info('Loading the model to the plugin')
    exec_net = ie.load_network(network=net, device_name=args.device)

# ---------------------------Step 5. Create infer request--------------------------------------------------------------
# load_network() method of the IECore class with a specified number of requests (default 1) returns an ExecutableNetwork
# instance which stores infer requests. So you already created Infer requests in the previous step.

# ---------------------------Step 6. Prepare input---------------------------------------------------------------------
    utterances = read_ark_file(args.input)
    if args.reference:
        scores = read_ark_file(args.reference)

# ---------------------------Step 7. Do inference----------------------------------------------------------------------
    log.info('Starting inference in synchronous mode')
    results = []
    infer_times = []

    for matrix in utterances:
        start_infer_time = default_timer()
        results.append(infer_matrix(matrix, exec_net, input_blob, out_blob))
        infer_times.append(default_timer() - start_infer_time)

# ---------------------------Step 8. Process output--------------------------------------------------------------------
    for i in range(len(utterances)):
        log.info(f'Utterance {i}')
        log.info(f'Frames in utterance: {utterances[i].shape[0]}')
        log.info(f'Total time in Infer (HW and SW): {infer_times[i] * 1000:.2f}ms')

        if args.reference:
            error_matrix = results[i] - scores[i]
            log.info(f'max error: {np.max(error_matrix)}')

        log.info('')

    log.info(f'Total sample time: {sum(infer_times) * 1000:.2f}ms')

# ----------------------------------------------------------------------------------------------------------------------
    log.info('This sample is an API example, '
             'for any performance measurements please use the dedicated benchmark_app tool\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
