#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import logging as log
import sys
import typing
from functools import reduce

import numpy as np
import openvino as ov
from openvino import op, opset1, opset8

from data import digits


def create_model(model_path: str) -> ov.Model:
    """Create a model on the fly from the source code using openvino."""

    def shape_and_length(shape: list) -> typing.Tuple[list, int]:
        length = reduce(lambda x, y: x * y, shape)
        return shape, length

    weights = np.fromfile(model_path, dtype=np.float32)
    weights_offset = 0
    padding_begin = padding_end = [0, 0]

    # input
    input_shape = [64, 1, 28, 28]
    param_node = op.Parameter(ov.Type.f32, ov.Shape(input_shape))

    # convolution 1
    conv_1_kernel_shape, conv_1_kernel_length = shape_and_length([20, 1, 5, 5])
    conv_1_kernel = op.Constant(ov.Type.f32, ov.Shape(conv_1_kernel_shape), weights[0:conv_1_kernel_length].tolist())
    weights_offset += conv_1_kernel_length
    conv_1_node = opset8.convolution(param_node, conv_1_kernel, [1, 1], padding_begin, padding_end, [1, 1])

    # add 1
    add_1_kernel_shape, add_1_kernel_length = shape_and_length([1, 20, 1, 1])
    add_1_kernel = op.Constant(ov.Type.f32, ov.Shape(add_1_kernel_shape),
                               weights[weights_offset : weights_offset + add_1_kernel_length])
    weights_offset += add_1_kernel_length
    add_1_node = opset8.add(conv_1_node, add_1_kernel)

    # maxpool 1
    maxpool_1_node = opset1.max_pool(add_1_node, [2, 2], padding_begin, padding_end, [2, 2], 'ceil')

    # convolution 2
    conv_2_kernel_shape, conv_2_kernel_length = shape_and_length([50, 20, 5, 5])
    conv_2_kernel = op.Constant(ov.Type.f32, ov.Shape(conv_2_kernel_shape),
                                weights[weights_offset : weights_offset + conv_2_kernel_length],
                                )
    weights_offset += conv_2_kernel_length
    conv_2_node = opset8.convolution(maxpool_1_node, conv_2_kernel, [1, 1], padding_begin, padding_end, [1, 1])

    # add 2
    add_2_kernel_shape, add_2_kernel_length = shape_and_length([1, 50, 1, 1])
    add_2_kernel = op.Constant(ov.Type.f32, ov.Shape(add_2_kernel_shape),
                               weights[weights_offset : weights_offset + add_2_kernel_length],
                               )
    weights_offset += add_2_kernel_length
    add_2_node = opset8.add(conv_2_node, add_2_kernel)

    # maxpool 2
    maxpool_2_node = opset1.max_pool(add_2_node, [2, 2], padding_begin, padding_end, [2, 2], 'ceil')

    # reshape 1
    reshape_1_dims, reshape_1_length = shape_and_length([2])
    # workaround to get int64 weights from float32 ndarray w/o unnecessary copying
    dtype_weights = np.frombuffer(
        weights[weights_offset : weights_offset + 2 * reshape_1_length],
        dtype=np.int64,
    )
    reshape_1_kernel = op.Constant(ov.Type.i64, ov.Shape(list(dtype_weights.shape)), dtype_weights)
    weights_offset += 2 * reshape_1_length
    reshape_1_node = opset8.reshape(maxpool_2_node, reshape_1_kernel, True)

    # matmul 1
    matmul_1_kernel_shape, matmul_1_kernel_length = shape_and_length([500, 800])
    matmul_1_kernel = op.Constant(ov.Type.f32, ov.Shape(matmul_1_kernel_shape),
                                  weights[weights_offset : weights_offset + matmul_1_kernel_length],
                                  )
    weights_offset += matmul_1_kernel_length
    matmul_1_node = opset8.matmul(reshape_1_node, matmul_1_kernel, False, True)

    # add 3
    add_3_kernel_shape, add_3_kernel_length = shape_and_length([1, 500])
    add_3_kernel = op.Constant(ov.Type.f32, ov.Shape(add_3_kernel_shape),
                               weights[weights_offset : weights_offset + add_3_kernel_length],
                               )
    weights_offset += add_3_kernel_length
    add_3_node = opset8.add(matmul_1_node, add_3_kernel)

    # ReLU
    relu_node = opset8.relu(add_3_node)

    # reshape 2
    reshape_2_kernel = op.Constant(ov.Type.i64, ov.Shape(list(dtype_weights.shape)), dtype_weights)
    reshape_2_node = opset8.reshape(relu_node, reshape_2_kernel, True)

    # matmul 2
    matmul_2_kernel_shape, matmul_2_kernel_length = shape_and_length([10, 500])
    matmul_2_kernel = op.Constant(ov.Type.f32, ov.Shape(matmul_2_kernel_shape),
                                  weights[weights_offset : weights_offset + matmul_2_kernel_length],
                                  )
    weights_offset += matmul_2_kernel_length
    matmul_2_node = opset8.matmul(reshape_2_node, matmul_2_kernel, False, True)

    # add 4
    add_4_kernel_shape, add_4_kernel_length = shape_and_length([1, 10])
    add_4_kernel = op.Constant(ov.Type.f32, ov.Shape(add_4_kernel_shape),
                               weights[weights_offset : weights_offset + add_4_kernel_length],
                               )
    weights_offset += add_4_kernel_length
    add_4_node = opset8.add(matmul_2_node, add_4_kernel)

    # softmax
    softmax_axis = 1
    softmax_node = opset8.softmax(add_4_node, softmax_axis)

    return ov.Model(softmax_node, [param_node], 'lenet')


def main():
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)
    # Parsing and validation of input arguments
    if len(sys.argv) != 3:
        log.info(f'Usage: {sys.argv[0]} <path_to_model> <device_name>')
        return 1

    model_path = sys.argv[1]
    device_name = sys.argv[2]
    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    number_top = 1
    # ---------------------------Step 1. Initialize OpenVINO Runtime Core--------------------------------------------------
    log.info('Creating OpenVINO Runtime Core')

    # ---------------------------Step 2. Read a model in OpenVINO Intermediate Representation------------------------------
    log.info(f'Loading the model using openvino with weights from {model_path}')
    model = create_model(model_path)
    # ---------------------------Step 3. Apply preprocessing----------------------------------------------------------
    # Get names of input and output blobs
    ppp = ov.preprocess.PrePostProcessor(model)
    # 1) Set input tensor information:
    # - input() provides information about a single model input
    # - precision of tensor is supposed to be 'u8'
    # - layout of data is 'NHWC'
    ppp.input().tensor() \
        .set_element_type(ov.Type.u8) \
        .set_layout(ov.Layout('NHWC'))  # noqa: N400

    # 2) Here we suppose model has 'NCHW' layout for input
    ppp.input().model().set_layout(ov.Layout('NCHW'))
    # 3) Set output tensor information:
    # - precision of tensor is supposed to be 'f32'
    ppp.output().tensor().set_element_type(ov.Type.f32)

    # 4) Apply preprocessing modifing the original 'model'
    model = ppp.build()

    # Set a batch size equal to number of input images
    ov.set_batch(model, digits.shape[0])
    # ---------------------------Step 4. Loading model to the device-------------------------------------------------------
    log.info('Loading the model to the plugin')
    core = ov.Core()
    compiled_model = core.compile_model(model, device_name)

    # ---------------------------Step 5. Prepare input---------------------------------------------------------------------
    n, c, h, w = model.input().shape
    input_data = np.ndarray(shape=(n, c, h, w))
    for i in range(n):
        image = digits[i].reshape(28, 28)
        image = image[:, :, np.newaxis]
        input_data[i] = image

    # ---------------------------Step 6. Do inference----------------------------------------------------------------------
    log.info('Starting inference in synchronous mode')
    results = compiled_model.infer_new_request({0: input_data})

    # ---------------------------Step 7. Process output--------------------------------------------------------------------
    predictions = next(iter(results.values()))

    log.info(f'Top {number_top} results: ')
    for i in range(n):
        probs = predictions[i]
        # Get an array of number_top class IDs in descending order of probability
        top_n_idexes = np.argsort(probs)[-number_top :][::-1]

        header = 'classid probability'
        header = header + ' label' if labels else header

        log.info(f'Image {i}')
        log.info('')
        log.info(header)
        log.info('-' * len(header))

        for class_id in top_n_idexes:
            probability_indent = ' ' * (len('classid') - len(str(class_id)) + 1)
            label_indent = ' ' * (len('probability') - 8) if labels else ''
            label = labels[class_id] if labels else ''
            log.info(f'{class_id}{probability_indent}{probs[class_id]:.7f}{label_indent}{label}')
        log.info('')

    # ----------------------------------------------------------------------------------------------------------------------
    log.info('This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
