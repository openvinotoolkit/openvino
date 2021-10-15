#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import argparse
import logging as log
import sys

import cv2
import numpy as np
from openvino.inference_engine import IECore, StatusCode


def parse_args() -> argparse.Namespace:
    """Parse and return command line arguments"""
    parser = argparse.ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    # fmt: off
    args.add_argument('-h', '--help', action='help', help='Show this help message and exit.')
    args.add_argument('-m', '--model', required=True, type=str,
                      help='Required. Path to an .xml or .onnx file with a trained model.')
    args.add_argument('-i', '--input', required=True, type=str, nargs='+', help='Required. Path to an image file(s).')
    args.add_argument('-l', '--extension', type=str, default=None,
                      help='Optional. Required by the CPU Plugin for executing the custom operation on a CPU. '
                      'Absolute path to a shared library with the kernels implementations.')
    args.add_argument('-c', '--config', type=str, default=None,
                      help='Optional. Required by GPU or VPU Plugins for the custom operation kernel. '
                      'Absolute path to operation description file (.xml).')
    args.add_argument('-d', '--device', default='CPU', type=str,
                      help='Optional. Specify the target device to infer on; CPU, GPU, MYRIAD, HDDL or HETERO: '
                      'is acceptable. The sample will look for a suitable plugin for device specified. '
                      'Default value is CPU.')
    args.add_argument('--labels', default=None, type=str, help='Optional. Path to a labels mapping file.')
    args.add_argument('-nt', '--number_top', default=10, type=int, help='Optional. Number of top results.')
    # fmt: on
    return parser.parse_args()


def main():
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)
    args = parse_args()

    # ---------------------------Step 1. Initialize inference engine core--------------------------------------------------
    log.info('Creating Inference Engine')
    ie = IECore()

    if args.extension and args.device == 'CPU':
        log.info(f'Loading the {args.device} extension: {args.extension}')
        ie.add_extension(args.extension, args.device)

    if args.config and args.device in ('GPU', 'MYRIAD', 'HDDL'):
        log.info(f'Loading the {args.device} configuration: {args.config}')
        ie.set_config({'CONFIG_FILE': args.config}, args.device)

    # ---------------------------Step 2. Read a model in OpenVINO Intermediate Representation or ONNX format---------------
    log.info(f'Reading the network: {args.model}')
    # (.xml and .bin files) or (.onnx file)
    net = ie.read_network(model=args.model)

    if len(net.input_info) != 1:
        log.error('Sample supports only single input topologies')
        return -1
    if len(net.outputs) != 1:
        log.error('Sample supports only single output topologies')
        return -1

    # ---------------------------Step 3. Configure input & output----------------------------------------------------------
    log.info('Configuring input and output blobs')
    # Get names of input and output blobs
    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))

    # Set input and output precision manually
    net.input_info[input_blob].precision = 'U8'
    net.outputs[out_blob].precision = 'FP32'

    # Get a number of input images
    num_of_input = len(args.input)
    # Get a number of classes recognized by a model
    num_of_classes = max(net.outputs[out_blob].shape)

    # ---------------------------Step 4. Loading model to the device-------------------------------------------------------
    log.info('Loading the model to the plugin')
    exec_net = ie.load_network(network=net, device_name=args.device, num_requests=num_of_input)

    # ---------------------------Step 5. Create infer request--------------------------------------------------------------
    # load_network() method of the IECore class with a specified number of requests (default 1) returns an ExecutableNetwork
    # instance which stores infer requests. So you already created Infer requests in the previous step.

    # ---------------------------Step 6. Prepare input---------------------------------------------------------------------
    input_data = []
    _, _, h, w = net.input_info[input_blob].input_data.shape

    for i in range(num_of_input):
        image = cv2.imread(args.input[i])

        if image.shape[:-1] != (h, w):
            log.warning(f'Image {args.input[i]} is resized from {image.shape[:-1]} to {(h, w)}')
            image = cv2.resize(image, (w, h))

        # Change data layout from HWC to CHW
        image = image.transpose((2, 0, 1))
        # Add N dimension to transform to NCHW
        image = np.expand_dims(image, axis=0)

        input_data.append(image)

    # ---------------------------Step 7. Do inference----------------------------------------------------------------------
    log.info('Starting inference in asynchronous mode')
    for i in range(num_of_input):
        exec_net.requests[i].async_infer({input_blob: input_data[i]})

    # ---------------------------Step 8. Process output--------------------------------------------------------------------
    # Generate a label list
    if args.labels:
        with open(args.labels, 'r') as f:
            labels = [line.split(',')[0].strip() for line in f]

    # Create a list to control a order of output
    output_queue = list(range(num_of_input))

    while True:
        for i in output_queue:
            # Immediately returns a inference status without blocking or interrupting
            infer_status = exec_net.requests[i].wait(0)

            if infer_status == StatusCode.RESULT_NOT_READY:
                continue

            log.info(f'Infer request {i} returned {infer_status}')

            if infer_status != StatusCode.OK:
                return -2

            # Read infer request results from buffer
            res = exec_net.requests[i].output_blobs[out_blob].buffer
            # Change a shape of a numpy.ndarray with results to get another one with one dimension
            probs = res.reshape(num_of_classes)
            # Get an array of args.number_top class IDs in descending order of probability
            top_n_idexes = np.argsort(probs)[-args.number_top :][::-1]

            header = 'classid probability'
            header = header + ' label' if args.labels else header

            log.info(f'Image path: {args.input[i]}')
            log.info(f'Top {args.number_top} results: ')
            log.info(header)
            log.info('-' * len(header))

            for class_id in top_n_idexes:
                probability_indent = ' ' * (len('classid') - len(str(class_id)) + 1)
                label_indent = ' ' * (len('probability') - 8) if args.labels else ''
                label = labels[class_id] if args.labels else ''
                log.info(f'{class_id}{probability_indent}{probs[class_id]:.7f}{label_indent}{label}')
            log.info('')

            output_queue.remove(i)

        if len(output_queue) == 0:
            break

    # ----------------------------------------------------------------------------------------------------------------------
    log.info('This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
