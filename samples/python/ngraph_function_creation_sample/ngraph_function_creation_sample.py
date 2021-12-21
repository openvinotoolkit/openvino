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
import numpy as np
from openvino.preprocess import PrePostProcessor
from openvino.runtime import Core, Layout, Type, Model, Shape, PartialShape
import openvino


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


def create_ngraph_function(args: argparse.Namespace) -> Model:
    """Create a network on the fly from the source code using ngraph"""

    def shape_and_length(shape: list) -> typing.Tuple[list, int]:
        length = reduce(lambda x, y: x * y, shape)
        return shape, length

    weights = np.fromfile(args.model, dtype=np.float32)
    weights_offset = 0
    padding_begin = padding_end = [0, 0]

    # input
    input_shape = [64, 1, 28, 28]
    param_node = openvino.runtime.op.Parameter(Type.f32, Shape(input_shape))

    # convolution 1
    conv_1_kernel_shape, conv_1_kernel_length = shape_and_length([20, 1, 5, 5])
    conv_1_kernel = openvino.runtime.op.Constant(Type.f32, Shape(conv_1_kernel_shape), weights[0:conv_1_kernel_length].tolist())
    weights_offset += conv_1_kernel_length
    conv_1_node = openvino.runtime.opset8.convolution(param_node, conv_1_kernel, [1, 1], padding_begin, padding_end, [1, 1])

    # add 1
    add_1_kernel_shape, add_1_kernel_length = shape_and_length([1, 20, 1, 1])
    add_1_kernel = openvino.runtime.op.Constant(Type.f32, Shape(add_1_kernel_shape),
                                                weights[weights_offset : weights_offset + add_1_kernel_length])
    weights_offset += add_1_kernel_length
    add_1_node = openvino.runtime.opset8.add(conv_1_node, add_1_kernel)

    # maxpool 1
    maxpool_1_node = openvino.runtime.opset1.max_pool(add_1_node, [2, 2], padding_begin, padding_end, [2, 2], 'ceil')

    # convolution 2
    conv_2_kernel_shape, conv_2_kernel_length = shape_and_length([50, 20, 5, 5])
    conv_2_kernel = openvino.runtime.op.Constant(Type.f32, Shape(conv_2_kernel_shape),
                                                 weights[weights_offset : weights_offset + conv_2_kernel_length]
                                                 )
    weights_offset += conv_2_kernel_length
    conv_2_node = openvino.runtime.opset8.convolution(maxpool_1_node, conv_2_kernel, [1, 1], padding_begin, padding_end, [1, 1])

    # add 2
    add_2_kernel_shape, add_2_kernel_length = shape_and_length([1, 50, 1, 1])
    add_2_kernel = openvino.runtime.op.Constant(Type.f32, Shape(add_2_kernel_shape),
                                                weights[weights_offset : weights_offset + add_2_kernel_length]
                                                )
    weights_offset += add_2_kernel_length
    add_2_node = openvino.runtime.opset8.add(conv_2_node, add_2_kernel)

    # maxpool 2
    maxpool_2_node = openvino.runtime.opset1.max_pool(add_2_node, [2, 2], padding_begin, padding_end, [2, 2], 'ceil')

    # reshape 1
    reshape_1_dims, reshape_1_length = shape_and_length([2])
    # workaround to get int64 weights from float32 ndarray w/o unnecessary copying
    dtype_weights = np.frombuffer(
        weights[weights_offset : weights_offset + 2 * reshape_1_length],
        dtype=np.int64,
    )
    reshape_1_kernel = openvino.runtime.op.Constant(Type.i64, Shape(list(dtype_weights.shape)), dtype_weights)
    weights_offset += 2 * reshape_1_length
    reshape_1_node = openvino.runtime.opset8.reshape(maxpool_2_node, reshape_1_kernel, True)

    # matmul 1
    matmul_1_kernel_shape, matmul_1_kernel_length = shape_and_length([500, 800])
    matmul_1_kernel = openvino.runtime.op.Constant(Type.f32, Shape(matmul_1_kernel_shape),
                                                   weights[weights_offset : weights_offset + matmul_1_kernel_length]
                                                   )
    weights_offset += matmul_1_kernel_length
    matmul_1_node = openvino.runtime.opset8.matmul(reshape_1_node, matmul_1_kernel, False, True)

    # add 3
    add_3_kernel_shape, add_3_kernel_length = shape_and_length([1, 500])
    add_3_kernel = openvino.runtime.op.Constant(Type.f32, Shape(add_3_kernel_shape),
                                                weights[weights_offset : weights_offset + add_3_kernel_length]
                                                )
    weights_offset += add_3_kernel_length
    add_3_node = openvino.runtime.opset8.add(matmul_1_node, add_3_kernel)

    # ReLU
    relu_node = openvino.runtime.opset8.relu(add_3_node)

    # reshape 2
    reshape_2_kernel = openvino.runtime.op.Constant(Type.i64, Shape(list(dtype_weights.shape)), dtype_weights)
    reshape_2_node = openvino.runtime.opset8.reshape(relu_node, reshape_2_kernel, True)

    # matmul 2
    matmul_2_kernel_shape, matmul_2_kernel_length = shape_and_length([10, 500])
    matmul_2_kernel = openvino.runtime.op.Constant(Type.f32, Shape(matmul_2_kernel_shape),
                                                   weights[weights_offset : weights_offset + matmul_2_kernel_length]
                                                   )
    weights_offset += matmul_2_kernel_length
    matmul_2_node = openvino.runtime.opset8.matmul(reshape_2_node, matmul_2_kernel, False, True)

    # add 4
    add_4_kernel_shape, add_4_kernel_length = shape_and_length([1, 10])
    add_4_kernel = openvino.runtime.op.Constant(Type.f32, Shape(add_4_kernel_shape),
                                                weights[weights_offset : weights_offset + add_4_kernel_length]
                                                )
    weights_offset += add_4_kernel_length
    add_4_node = openvino.runtime.opset8.add(matmul_2_node, add_4_kernel)

    # softmax
    softmax_axis = 1
    softmax_node = openvino.runtime.opset8.softmax(add_4_node, softmax_axis)

    # result
    result_node = openvino.runtime.opset8.result(softmax_node)

    return Model(softmax_node, [param_node], 'lenet')


def main():
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)
    args = parse_args()

    # ---------------------------Step 1. Initialize inference engine core--------------------------------------------------
    log.info('Creating OpenVINO Runtime Core')
    core = Core()

    # ---------------------------Step 2. Read a model in OpenVINO Intermediate Representation------------------------------
    log.info(f'Loading the network using ngraph function with weights from {args.model}')
    model = create_ngraph_function(args)
    # ---------------------------Step 3. Apply preprocessing----------------------------------------------------------
    # Get names of input and output blobs
    ppp = PrePostProcessor(model)
    # 1) Set input tensor information:
    # - input() provides information about a single model input
    # - precision of tensor is supposed to be 'u8'
    # - layout of data is 'NHWC'
    ppp.input().tensor() \
        .set_element_type(Type.u8) \
        .set_layout(Layout('NHWC'))  # noqa: N400

    # 2) Here we suppose model has 'NCHW' layout for input
    ppp.input().model().set_layout(Layout('NCHW'))
    # 3) Set output tensor information:
    # - precision of tensor is supposed to be 'f32'
    ppp.output().tensor().set_element_type(Type.f32)

    # 4) Apply preprocessing modifing the original 'model'
    model = ppp.build()

    # Set a batch size to a equal number of input images
    model.reshape({model.input().get_any_name(): PartialShape((len(args.input), model.input().shape[1], model.input().shape[2], model.input().shape[3]))})

    # ---------------------------Step 4. Loading model to the device-------------------------------------------------------
    log.info('Loading the model to the plugin')
    compiled_model = core.compile_model(model, args.device)

    # ---------------------------Step 5. Prepare input---------------------------------------------------------------------
    n, c, h, w = model.input().shape
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

    # ---------------------------Step 6. Do inference----------------------------------------------------------------------
    log.info('Starting inference in synchronous mode')
    results = compiled_model.infer_new_request({0: input_data})

    # ---------------------------Step 7. Process output--------------------------------------------------------------------
    # Generate a label list
    if args.labels:
        with open(args.labels, 'r') as f:
            labels = [line.split(',')[0].strip() for line in f]

    predictions = next(iter(results.values()))

    for i in range(n):
        probs = predictions[i]
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
