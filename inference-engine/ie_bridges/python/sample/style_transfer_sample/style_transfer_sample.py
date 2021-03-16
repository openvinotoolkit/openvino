#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import argparse
import logging as log
import sys

import cv2
import numpy as np
from openvino.inference_engine import IECore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
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
    args.add_argument('--mean_val_r', default=0, type=float,
                      help='Optional. Mean value of red channel for mean value subtraction in postprocessing.')
    args.add_argument('--mean_val_g', default=0, type=float,
                      help='Optional. Mean value of green channel for mean value subtraction in postprocessing.')
    args.add_argument('--mean_val_b', default=0, type=float,
                      help='Optional. Mean value of blue channel for mean value subtraction in postprocessing.')

    return parser.parse_args()


def main():
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO)
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

# ---------------------------Step 2. Read a model in OpenVINO Intermediate Representation------------------------------
    log.info(f'Reading the network: {args.model}')
    # (.xml and .bin files) or ONNX (.onnx file) format
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

# ---------------------------Step 4. Loading model to the device-------------------------------------------------------
    log.info('Loading the model to the plugin')
    exec_net = ie.load_network(network=net, device_name=args.device)

# ---------------------------Step 5. Create infer request--------------------------------------------------------------
# To make a valid InferRequest instance, use load_network() method of the IECore class with specified number of
# requests (1 by default) to get ExecutableNetwork instance which stores infer requests.

# ---------------------------Step 6. Prepare input---------------------------------------------------------------------
    original_image = cv2.imread(args.input)
    image = original_image.copy()
    _, _, h, w = net.input_info[input_blob].input_data.shape

    if image.shape[:-1] != (h, w):
        log.warning(f'Image {args.input} is resized from {image.shape[:-1]} to {(h, w)}')
        image = cv2.resize(image, (w, h))

    # Change data layout from HWC to CHW
    image = image.transpose((2, 0, 1))
    # Add N dimension to transform to NCHW
    image = np.expand_dims(image, axis=0)

# ---------------------------Step 7. Do inference----------------------------------------------------------------------
    log.info('Starting inference in synchronous mode')
    res = exec_net.infer(inputs={input_blob: image})

# ---------------------------Step 8. Process output--------------------------------------------------------------------
    res = res[out_blob]
    output_image = res[0]
    # Change data layout from CHW to HWC
    output_image = output_image.transpose((1, 2, 0))
    # Convert BGR color order to RGB
    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

    # Apply mean argument values
    output_image = output_image[::] - (args.mean_val_r, args.mean_val_g, args.mean_val_b)
    # Set pixel values bitween 0 and 255
    output_image = np.clip(output_image, 0, 255)

    # Resize a output image to original size
    h, w, _ = original_image.shape
    output_image = cv2.resize(output_image, (w, h))

    cv2.imwrite('out.bmp', output_image)
    log.info('Image out.bmp created!')

# ----------------------------------------------------------------------------------------------------------------------
    log.info('This sample is an API example, '
             'for any performance measurements please use the dedicated benchmark_app tool\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
