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


def main():  # noqa
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
        log.error('The sample supports only single input topologies')
        return -1

    if len(net.outputs) != 1 and not ('boxes' in net.outputs or 'labels' in net.outputs):
        log.error('The sample supports models with 1 output or with 2 with the names "boxes" and "labels"')
        return -1

    # ---------------------------Step 3. Configure input & output----------------------------------------------------------
    log.info('Configuring input and output blobs')
    # Get name of input blob
    input_blob = next(iter(net.input_info))

    # Set input and output precision manually
    net.input_info[input_blob].precision = 'U8'

    if len(net.outputs) == 1:
        output_blob = next(iter(net.outputs))
        net.outputs[output_blob].precision = 'FP32'
    else:
        net.outputs['boxes'].precision = 'FP32'
        net.outputs['labels'].precision = 'U16'

    # ---------------------------Step 4. Loading model to the device-------------------------------------------------------
    log.info('Loading the model to the plugin')
    exec_net = ie.load_network(network=net, device_name=args.device)

    # ---------------------------Step 5. Create infer request--------------------------------------------------------------
    # load_network() method of the IECore class with a specified number of requests (default 1) returns an ExecutableNetwork
    # instance which stores infer requests. So you already created Infer requests in the previous step.

    # ---------------------------Step 6. Prepare input---------------------------------------------------------------------
    original_image = cv2.imread(args.input)
    image = original_image.copy()
    _, _, net_h, net_w = net.input_info[input_blob].input_data.shape

    if image.shape[:-1] != (net_h, net_w):
        log.warning(f'Image {args.input} is resized from {image.shape[:-1]} to {(net_h, net_w)}')
        image = cv2.resize(image, (net_w, net_h))

    # Change data layout from HWC to CHW
    image = image.transpose((2, 0, 1))
    # Add N dimension to transform to NCHW
    image = np.expand_dims(image, axis=0)

    # ---------------------------Step 7. Do inference----------------------------------------------------------------------
    log.info('Starting inference in synchronous mode')
    res = exec_net.infer(inputs={input_blob: image})

    # ---------------------------Step 8. Process output--------------------------------------------------------------------
    # Generate a label list
    if args.labels:
        with open(args.labels, 'r') as f:
            labels = [line.split(',')[0].strip() for line in f]

    output_image = original_image.copy()
    h, w, _ = output_image.shape

    if len(net.outputs) == 1:
        res = res[output_blob]
        # Change a shape of a numpy.ndarray with results ([1, 1, N, 7]) to get another one ([N, 7]),
        # where N is the number of detected bounding boxes
        detections = res.reshape(-1, 7)
    else:
        detections = res['boxes']
        labels = res['labels']
        # Redefine scale coefficients
        w, h = w / net_w, h / net_h

    for i, detection in enumerate(detections):
        if len(net.outputs) == 1:
            _, class_id, confidence, xmin, ymin, xmax, ymax = detection
        else:
            class_id = labels[i]
            xmin, ymin, xmax, ymax, confidence = detection

        if confidence > 0.5:
            label = int(labels[class_id]) if args.labels else int(class_id)

            xmin = int(xmin * w)
            ymin = int(ymin * h)
            xmax = int(xmax * w)
            ymax = int(ymax * h)

            log.info(f'Found: label = {label}, confidence = {confidence:.2f}, ' f'coords = ({xmin}, {ymin}), ({xmax}, {ymax})')

            # Draw a bounding box on a output image
            cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    cv2.imwrite('out.bmp', output_image)
    log.info('Image out.bmp created!')

    # ----------------------------------------------------------------------------------------------------------------------
    log.info('This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
