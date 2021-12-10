#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import argparse
import logging as log
import sys

import cv2
import numpy as np
from openvino.runtime import Core, Tensor, Layout, Type
from openvino.preprocess import PrePostProcessor, ResizeAlgorithm


def parse_args() -> argparse.Namespace:
    """Parse and return command line arguments"""
    parser = argparse.ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    # fmt: off
    args.add_argument('-h', '--help', action='help', help='Show this help message and exit.')
    args.add_argument('-m', '--model', required=True, type=str,
                      help='Required. Path to an .xml or .onnx file with a trained model.')
    args.add_argument('-i', '--input', required=True, type=str, help='Required. Path to an image file.')
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

    # ---------------------------Step 1. Initialize OpenVINO Runtime Core--------------------------------------------------
    log.info('Creating OpenVINO Runtime Core')
    core = Core()

    # ---------------------------Step 2. Read a model in OpenVINO Intermediate Representation or ONNX format---------------
    log.info(f'Reading the network: {args.model}')
    # (.xml and .bin files) or (.onnx file)

    model = core.read_model(model=args.model)

    # ---------------------------Step 3. Configure image_tensor----------------------------------------------------------

    if len(model.inputs) != 1:
        log.error('Sample supports only single input topologies')
        return -1
    if model.get_output_size() != 1:
        log.error('Sample supports only single output topologies')
        return -1

    # Read input image
    image = cv2.imread(args.input)
    # Add N dimension to transform to NCHW
    image = np.expand_dims(image, axis=0)
    # Wrap input image into openvino.runtime.Tensor without allocating of new memory
    image_tensor = Tensor(image)
    tensor_layout = Layout('NHWC')
    h, w = image_tensor.shape[1], image_tensor.shape[2]

    # ---------------------------Step 4. Apply preprocessing----------------------------------------------------------
    preproc = PrePostProcessor(model)
    # 1) Set input tensor information:
    # - input() provides information about a single model input
    # - precision of tensor is supposed to be 'u8'
    # - layout of data is 'NHWC'
    # - set static spatial dimensions to input tensor to resize from
    preproc.input().tensor() \
           .set_element_type(image_tensor.element_type) \
           .set_layout(tensor_layout) \
           .set_spatial_static_shape(h, w) # noqa: ECE001, N400
    # 2) Adding explicit preprocessing steps:
    # - convert layout to 'NCHW' (from 'NHWC' specified above at tensor layout)
    # - apply linear resize from tensor spatial dims to model spatial dims
    preproc.input().preprocess().resize(ResizeAlgorithm.RESIZE_LINEAR)
    # 3) Here we suppose model has 'NCHW' layout for input
    preproc.input().network().set_layout(Layout('NCHW'))
    # 4) Set output tensor information:
    # - precision of tensor is supposed to be 'f32'
    preproc.output().tensor().set_element_type(Type.f32)
    # 5) Apply preprocessing modifing the original 'model'
    model = preproc.build()

    # Get a number of classes recognized by a model
    num_of_classes = max(model.output().shape)

    # ---------------------------Step 5. Loading model to the device-------------------------------------------------------
    log.info('Loading the model to the plugin')
    compiled_model = core.compile_model(model=model, device_name=args.device)

    # ---------------------------Step 6. Create infer request--------------------------------------------------------------
    log.info('Starting inference in synchronous mode')

    res = compiled_model.infer_new_request(inputs={0: image_tensor})

    # ---------------------------Step 7. Process output--------------------------------------------------------------------
    # Generate a label list
    if args.labels:
        with open(args.labels, 'r') as f:
            labels = [line.split(',')[0].strip() for line in f]

    predictions = next(iter(res.values()))

    # Change a shape of a numpy.ndarray with results to get another one with one dimension
    probs = predictions.reshape(num_of_classes)
    # Get an array of args.number_top class IDs in descending order of probability
    top_n_idexes = np.argsort(probs)[-args.number_top :][::-1]

    header = 'classid probability'
    header = header + ' label' if args.labels else header

    log.info(f'Image path: {args.input}')
    log.info(f'Top {args.number_top} results: ')
    log.info(header)
    log.info('-' * len(header))

    for class_id in top_n_idexes:
        probability_indent = ' ' * (len('classid') - len(str(class_id)) + 1)
        label_indent = ' ' * (len('probability') - 8) if args.labels else ''
        label = labels[class_id] if args.labels else ''
        log.info(f'{class_id}{probability_indent}{probs[class_id]:.7f}{label_indent}{label}')
    log.info('')

    # ----------------------------------------------------------------------------------------------------------------------
    log.info('This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
