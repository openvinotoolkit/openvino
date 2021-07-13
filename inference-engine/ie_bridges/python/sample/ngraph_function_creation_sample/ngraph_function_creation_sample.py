#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import argparse
import logging as log
import struct as st
import sys
import typing
from functools import reduce

import cv2
import ngraph
import numpy as np
from openvino.inference_engine import IECore, IENetwork


def parse_args() -> argparse.Namespace:
    """Parse and return command line arguments"""
    parser = argparse.ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    # fmt: off
    args.add_argument('-h', '--help', action='help', help='Show this help message and exit.')
    args.add_argument('-m', '--model', required=True, type=str,
                      help='Required. Path to a file with network weights.')
    args.add_argument('-i', '--input', required=True, type=str, nargs='+', help='Required. Path to an image file.')
    args.add_argument('-d', '--device', default='CPU', type=str,
                      help='Optional. Specify the target device to infer on; CPU, GPU, MYRIAD, HDDL or HETERO: '
                      'is acceptable. The sample will look for a suitable plugin for device specified. '
                      'Default value is CPU.')
    args.add_argument('--labels', default=None, type=str, help='Optional. Path to a labels mapping file.')
    args.add_argument('-nt', '--number_top', default=10, type=int, help='Optional. Number of top results.')
    # fmt: on
    return parser.parse_args()


def read_image(image_path: str) -> np.ndarray:
    """Read and return an image as grayscale (one channel)"""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Try to open image as ubyte
    if image is None:
        with open(image_path, 'rb') as f:
            st.unpack('>4B', f.read(4))  # need to skip  4 bytes
            nimg = st.unpack('>I', f.read(4))[0]  # number of images
            nrow = st.unpack('>I', f.read(4))[0]  # number of rows
            ncolumn = st.unpack('>I', f.read(4))[0]  # number of column
            nbytes = nimg * nrow * ncolumn * 1  # each pixel data is 1 byte

            if nimg != 1:
                raise Exception('Sample supports ubyte files with 1 image inside')

            image = np.asarray(st.unpack('>' + 'B' * nbytes, f.read(nbytes))).reshape((nrow, ncolumn))

    return image


def create_ngraph_function(args: argparse.Namespace) -> ngraph.impl.Function:
    """Create a network on the fly from the source code using ngraph"""

    def shape_and_length(shape: list) -> typing.Tuple[list, int]:
        length = reduce(lambda x, y: x * y, shape)
        return shape, length

    weights = np.fromfile(args.model, dtype=np.float32)
    weights_offset = 0
    padding_begin = padding_end = [0, 0]

    # input
    input_shape = [64, 1, 28, 28]
    param_node = ngraph.parameter(input_shape, np.float32, 'Parameter')

    # convolution 1
    conv_1_kernel_shape, conv_1_kernel_length = shape_and_length([20, 1, 5, 5])
    conv_1_kernel = ngraph.constant(weights[0:conv_1_kernel_length].reshape(conv_1_kernel_shape))
    weights_offset += conv_1_kernel_length
    conv_1_node = ngraph.convolution(param_node, conv_1_kernel, [1, 1], padding_begin, padding_end, [1, 1])

    # add 1
    add_1_kernel_shape, add_1_kernel_length = shape_and_length([1, 20, 1, 1])
    add_1_kernel = ngraph.constant(
        weights[weights_offset : weights_offset + add_1_kernel_length].reshape(add_1_kernel_shape),
    )
    weights_offset += add_1_kernel_length
    add_1_node = ngraph.add(conv_1_node, add_1_kernel)

    # maxpool 1
    maxpool_1_node = ngraph.max_pool(add_1_node, [2, 2], padding_begin, padding_end, [2, 2], 'ceil', None)

    # convolution 2
    conv_2_kernel_shape, conv_2_kernel_length = shape_and_length([50, 20, 5, 5])
    conv_2_kernel = ngraph.constant(
        weights[weights_offset : weights_offset + conv_2_kernel_length].reshape(conv_2_kernel_shape),
    )
    weights_offset += conv_2_kernel_length
    conv_2_node = ngraph.convolution(maxpool_1_node, conv_2_kernel, [1, 1], padding_begin, padding_end, [1, 1])

    # add 2
    add_2_kernel_shape, add_2_kernel_length = shape_and_length([1, 50, 1, 1])
    add_2_kernel = ngraph.constant(
        weights[weights_offset : weights_offset + add_2_kernel_length].reshape(add_2_kernel_shape),
    )
    weights_offset += add_2_kernel_length
    add_2_node = ngraph.add(conv_2_node, add_2_kernel)

    # maxpool 2
    maxpool_2_node = ngraph.max_pool(add_2_node, [2, 2], padding_begin, padding_end, [2, 2], 'ceil', None)

    # reshape 1
    reshape_1_dims, reshape_1_length = shape_and_length([2])
    # workaround to get int64 weights from float32 ndarray w/o unnecessary copying
    dtype_weights = np.frombuffer(
        weights[weights_offset : weights_offset + 2 * reshape_1_length],
        dtype=np.int64,
    )
    reshape_1_kernel = ngraph.constant(dtype_weights)
    weights_offset += 2 * reshape_1_length
    reshape_1_node = ngraph.reshape(maxpool_2_node, reshape_1_kernel, True)

    # matmul 1
    matmul_1_kernel_shape, matmul_1_kernel_length = shape_and_length([500, 800])
    matmul_1_kernel = ngraph.constant(
        weights[weights_offset : weights_offset + matmul_1_kernel_length].reshape(matmul_1_kernel_shape),
    )
    weights_offset += matmul_1_kernel_length
    matmul_1_node = ngraph.matmul(reshape_1_node, matmul_1_kernel, False, True)

    # add 3
    add_3_kernel_shape, add_3_kernel_length = shape_and_length([1, 500])
    add_3_kernel = ngraph.constant(
        weights[weights_offset : weights_offset + add_3_kernel_length].reshape(add_3_kernel_shape),
    )
    weights_offset += add_3_kernel_length
    add_3_node = ngraph.add(matmul_1_node, add_3_kernel)

    # ReLU
    relu_node = ngraph.relu(add_3_node)

    # reshape 2
    reshape_2_kernel = ngraph.constant(dtype_weights)
    reshape_2_node = ngraph.reshape(relu_node, reshape_2_kernel, True)

    # matmul 2
    matmul_2_kernel_shape, matmul_2_kernel_length = shape_and_length([10, 500])
    matmul_2_kernel = ngraph.constant(
        weights[weights_offset : weights_offset + matmul_2_kernel_length].reshape(matmul_2_kernel_shape),
    )
    weights_offset += matmul_2_kernel_length
    matmul_2_node = ngraph.matmul(reshape_2_node, matmul_2_kernel, False, True)

    # add 4
    add_4_kernel_shape, add_4_kernel_length = shape_and_length([1, 10])
    add_4_kernel = ngraph.constant(
        weights[weights_offset : weights_offset + add_4_kernel_length].reshape(add_4_kernel_shape),
    )
    weights_offset += add_4_kernel_length
    add_4_node = ngraph.add(matmul_2_node, add_4_kernel)

    # softmax
    softmax_axis = 1
    softmax_node = ngraph.softmax(add_4_node, softmax_axis)

    # result
    result_node = ngraph.result(softmax_node)
    return ngraph.impl.Function(result_node, [param_node], 'lenet')


def main():
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)
    args = parse_args()

    # ---------------------------Step 1. Initialize inference engine core--------------------------------------------------
    log.info('Creating Inference Engine')
    ie = IECore()

    # ---------------------------Step 2. Read a model in OpenVINO Intermediate Representation------------------------------
    log.info(f'Loading the network using ngraph function with weights from {args.model}')
    ngraph_function = create_ngraph_function(args)
    net = IENetwork(ngraph.impl.Function.to_capsule(ngraph_function))

    # ---------------------------Step 3. Configure input & output----------------------------------------------------------
    log.info('Configuring input and output blobs')
    # Get names of input and output blobs
    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))

    # Set input and output precision manually
    net.input_info[input_blob].precision = 'U8'
    net.outputs[out_blob].precision = 'FP32'

    # Set a batch size to a equal number of input images
    net.batch_size = len(args.input)

    # ---------------------------Step 4. Loading model to the device-------------------------------------------------------
    log.info('Loading the model to the plugin')
    exec_net = ie.load_network(network=net, device_name=args.device)

    # ---------------------------Step 5. Create infer request--------------------------------------------------------------
    # load_network() method of the IECore class with a specified number of requests (default 1) returns an ExecutableNetwork
    # instance which stores infer requests. So you already created Infer requests in the previous step.

    # ---------------------------Step 6. Prepare input---------------------------------------------------------------------
    n, c, h, w = net.input_info[input_blob].input_data.shape
    input_data = np.ndarray(shape=(n, c, h, w))

    for i in range(n):
        image = read_image(args.input[i])

        light_pixel_count = np.count_nonzero(image > 127)
        dark_pixel_count = np.count_nonzero(image < 127)
        is_light_image = (light_pixel_count - dark_pixel_count) > 0

        if is_light_image:
            log.warning(f'Image {args.input[i]} is inverted to white over black')
            image = cv2.bitwise_not(image)

        if image.shape != (h, w):
            log.warning(f'Image {args.input[i]} is resized from {image.shape} to {(h, w)}')
            image = cv2.resize(image, (w, h))

        input_data[i] = image

    # ---------------------------Step 7. Do inference----------------------------------------------------------------------
    log.info('Starting inference in synchronous mode')
    res = exec_net.infer(inputs={input_blob: input_data})

    # ---------------------------Step 8. Process output--------------------------------------------------------------------
    # Generate a label list
    if args.labels:
        with open(args.labels, 'r') as f:
            labels = [line.split(',')[0].strip() for line in f]

    res = res[out_blob]

    for i in range(n):
        probs = res[i]
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

    # ----------------------------------------------------------------------------------------------------------------------
    log.info('This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
