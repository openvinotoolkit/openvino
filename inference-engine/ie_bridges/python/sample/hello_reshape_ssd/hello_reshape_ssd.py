#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import argparse
import logging as log
import sys

import cv2
import numpy as np
from openvino.inference_engine import IECore


def parse_args() -> argparse.Namespace:
    """Parse and return command line arguments"""
    parser = argparse.ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    # fmt: off
    args.add_argument('-h', '--help', action='help', help='Show this help message and exit.')
    args.add_argument('-m', '--model', required=True, type=str,
                      help='Required. Path to an .xml or .onnx file with a trained model.')
    args.add_argument('-i', '--input', required=True, type=str, help='Required. Path to an image file.')
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

    original_image = cv2.imread(args.input)
    image = original_image.copy()

    # Change data layout from HWC to CHW
    image = image.transpose((2, 0, 1))
    # Add N dimension to transform to NCHW
    image = np.expand_dims(image, axis=0)

    log.info('Reshaping the network to the height and width of the input image')
    log.info(f'Input shape before reshape: {net.input_info[input_blob].input_data.shape}')
    net.reshape({input_blob: image.shape})
    log.info(f'Input shape after reshape: {net.input_info[input_blob].input_data.shape}')

    # ---------------------------Step 4. Loading model to the device-------------------------------------------------------
    log.info('Loading the model to the plugin')
    exec_net = ie.load_network(network=net, device_name=args.device)

    # ---------------------------Step 5. Create infer request--------------------------------------------------------------
    # load_network() method of the IECore class with a specified number of requests (default 1) returns an ExecutableNetwork
    # instance which stores infer requests. So you already created Infer requests in the previous step.

    # ---------------------------Step 6. Prepare input---------------------------------------------------------------------
    # This sample changes a network input layer shape instead of a image shape. See Step 4.

    # ---------------------------Step 7. Do inference----------------------------------------------------------------------
    log.info('Starting inference in synchronous mode')
    res = exec_net.infer(inputs={input_blob: image})

    # ---------------------------Step 8. Process output--------------------------------------------------------------------
    # Generate a label list
    if args.labels:
        with open(args.labels, 'r') as f:
            labels = [line.split(',')[0].strip() for line in f]

    res = res[out_blob]

    output_image = original_image.copy()
    h, w, _ = output_image.shape

    # Change a shape of a numpy.ndarray with results ([1, 1, N, 7]) to get another one ([N, 7]),
    # where N is the number of detected bounding boxes
    detections = res.reshape(-1, 7)

    for detection in detections:
        confidence = detection[2]
        if confidence > 0.5:
            class_id = int(detection[1])
            label = labels[class_id] if args.labels else class_id

            xmin = int(detection[3] * w)
            ymin = int(detection[4] * h)
            xmax = int(detection[5] * w)
            ymax = int(detection[6] * h)

            log.info(f'Found: label = {label}, confidence = {confidence:.2f}, ' f'coords = ({xmin}, {ymin}), ({xmax}, {ymax})')

            # Draw a bounding box on a output image
            cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    cv2.imwrite('out.bmp', output_image)
    log.info('Image out.bmp was created!')

    # ----------------------------------------------------------------------------------------------------------------------
    log.info('This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
