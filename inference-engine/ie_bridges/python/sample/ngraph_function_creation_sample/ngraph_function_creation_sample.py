#!/usr/bin/env python
"""
 Copyright (C) 2018-2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import sys
import os
from argparse import ArgumentParser, SUPPRESS
import cv2
import numpy as np
import logging as log
from openvino.inference_engine import IECore, IENetwork
import ngraph
from ngraph.impl import Function
from functools import reduce
import struct as st


def build_argparser() -> ArgumentParser:
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument('-i', '--input', help='Required. Path to a folder with images or path to an image files',
                      required=True,
                      type=str, nargs="+")
    args.add_argument('-m', '--model', help='Required. Path to file where weights for the network are located')
    args.add_argument('-d', '--device',
                      help='Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL, MYRIAD or HETERO: '
                           'is acceptable. The sample will look for a suitable plugin for device specified. Default '
                           'value is CPU',
                      default='CPU', type=str)
    args.add_argument('--labels', help='Optional. Path to a labels mapping file', default=None, type=str)
    args.add_argument('-nt', '--number_top', help='Optional. Number of top results', default=1, type=int)

    return parser


def list_input_images(input_dirs: list):
    images = []
    for input_dir in input_dirs:
        if os.path.isdir(input_dir):
            for root, directories, filenames in os.walk(input_dir):
                for filename in filenames:
                    images.append(os.path.join(root, filename))
        elif os.path.isfile(input_dir):
            images.append(input_dir)

    return images


def read_image(image_path: np):
    # try to read image in usual image formats
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # try to open image as ubyte
    if image is None:
        with open(image_path, 'rb') as f:
            image_file = open(image_path, 'rb')
            image_file.seek(0)
            st.unpack('>4B', image_file.read(4))    # need to skip  4 bytes
            nimg = st.unpack('>I', image_file.read(4))[0]  # number of images
            nrow = st.unpack('>I', image_file.read(4))[0]  # number of rows
            ncolumn = st.unpack('>I', image_file.read(4))[0]  # number of column
            nbytes = nimg * nrow * ncolumn * 1  # each pixel data is 1 byte

            assert nimg == 1, log.error('Sample supports ubyte files with 1 image inside')

            image = np.asarray(st.unpack('>' + 'B' * nbytes, image_file.read(nbytes))).reshape(
                (nrow, ncolumn))

    return image


def shape_and_length(shape: list):
    length = reduce(lambda x, y: x*y, shape)
    return shape, length


def create_ngraph_function(args) -> Function:
    weights = np.fromfile(args.model, dtype=np.float32)
    weights_offset = 0
    padding_begin = [0, 0]
    padding_end = [0, 0]

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
        weights[weights_offset:weights_offset + add_1_kernel_length].reshape(add_1_kernel_shape)
    )
    weights_offset += add_1_kernel_length
    add_1_node = ngraph.add(conv_1_node, add_1_kernel)

    # maxpool 1
    maxpool_1_node = ngraph.max_pool(add_1_node, [2, 2], padding_begin, padding_end, [2, 2], 'ceil', None)

    # convolution 2
    conv_2_kernel_shape, conv_2_kernel_length = shape_and_length([50, 20, 5, 5])
    conv_2_kernel = ngraph.constant(
        weights[weights_offset:weights_offset + conv_2_kernel_length].reshape(conv_2_kernel_shape)
    )
    weights_offset += conv_2_kernel_length
    conv_2_node = ngraph.convolution(maxpool_1_node, conv_2_kernel, [1, 1], padding_begin, padding_end, [1, 1])

    # add 2
    add_2_kernel_shape, add_2_kernel_length = shape_and_length([1, 50, 1, 1])
    add_2_kernel = ngraph.constant(
        weights[weights_offset:weights_offset + add_2_kernel_length].reshape(add_2_kernel_shape)
    )
    weights_offset += add_2_kernel_length
    add_2_node = ngraph.add(conv_2_node, add_2_kernel)

    # maxpool 2
    maxpool_2_node = ngraph.max_pool(add_2_node, [2, 2], padding_begin, padding_end, [2, 2], 'ceil', None)

    # reshape 1
    reshape_1_dims, reshape_1_length = shape_and_length([2])
    # workaround to get int64 weights from float32 ndarray w/o unnecessary copying
    dtype_weights = np.frombuffer(
        weights[weights_offset:weights_offset + 2*reshape_1_length], dtype=np.int64
    )
    reshape_1_kernel = ngraph.constant(dtype_weights)
    weights_offset += 2*reshape_1_length
    reshape_1_node = ngraph.reshape(maxpool_2_node, reshape_1_kernel, True)

    # matmul 1
    matmul_1_kernel_shape, matmul_1_kernel_length = shape_and_length([500, 800])
    matmul_1_kernel = ngraph.constant(
        weights[weights_offset:weights_offset + matmul_1_kernel_length].reshape(matmul_1_kernel_shape)
    )
    weights_offset += matmul_1_kernel_length
    matmul_1_node = ngraph.matmul(reshape_1_node, matmul_1_kernel, False, True)

    # add 3
    add_3_kernel_shape, add_3_kernel_length = shape_and_length([1, 500])
    add_3_kernel = ngraph.constant(
        weights[weights_offset:weights_offset + add_3_kernel_length].reshape(add_3_kernel_shape)
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
        weights[weights_offset:weights_offset + matmul_2_kernel_length].reshape(matmul_2_kernel_shape)
    )
    weights_offset += matmul_2_kernel_length
    matmul_2_node = ngraph.matmul(reshape_2_node, matmul_2_kernel, False, True)

    # add 4
    add_4_kernel_shape, add_4_kernel_length = shape_and_length([1, 10])
    add_4_kernel = ngraph.constant(
        weights[weights_offset:weights_offset + add_4_kernel_length].reshape(add_4_kernel_shape)
    )
    weights_offset += add_4_kernel_length
    add_4_node = ngraph.add(matmul_2_node, add_4_kernel)

    # softmax
    softmax_axis = 1
    softmax_node = ngraph.softmax(add_4_node, softmax_axis)

    # result
    result_node = ngraph.result(softmax_node)

    # nGraph function
    function = Function(result_node, [param_node], 'lenet')

    return function


def main():
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()

    input_images = list_input_images(args.input)

    # Loading network using ngraph function
    ngraph_function = create_ngraph_function(args)
    net = IENetwork(Function.to_capsule(ngraph_function))

    assert len(net.input_info.keys()) == 1, "Sample supports only single input topologies"
    assert len(net.outputs) == 1, "Sample supports only single output topologies"

    log.info("Preparing input blobs")
    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))
    net.batch_size = len(input_images)

    # Read and pre-process input images
    n, c, h, w = net.input_info[input_blob].input_data.shape
    images = np.ndarray(shape=(n, c, h, w))
    for i in range(n):
        image = read_image(input_images[i])
        assert image is not None, log.error("Can't open an image {}".format(input_images[i]))
        assert len(image.shape) == 2, log.error('Sample supports images with 1 channel only')
        if image.shape[:] != (w, h):
            log.warning("Image {} is resized from {} to {}".format(input_images[i], image.shape[:], (w, h)))
            image = cv2.resize(image, (w, h))
        images[i] = image
    log.info("Batch size is {}".format(n))

    log.info("Creating Inference Engine")
    ie = IECore()

    log.info('Loading model to the device')
    exec_net = ie.load_network(network=net, device_name=args.device.upper())

    # Start sync inference
    log.info('Creating infer request and starting inference')
    res = exec_net.infer(inputs={input_blob: images})

    # Processing results
    log.info("Processing output blob")
    res = res[out_blob]
    log.info("Top {} results: ".format(args.number_top))

    # Read labels file if it is provided as argument
    labels_map = None
    if args.labels:
        with open(args.labels, 'r') as f:
            labels_map = [x.split(sep=' ', maxsplit=1)[-1].strip() for x in f]

    classid_str = "classid"
    probability_str = "probability"
    for i, probs in enumerate(res):
        probs = np.squeeze(probs)
        top_ind = np.argsort(probs)[-args.number_top:][::-1]
        print("Image {}\n".format(input_images[i]))
        print(classid_str, probability_str)
        print("{} {}".format('-' * len(classid_str), '-' * len(probability_str)))
        for class_id in top_ind:
            det_label = labels_map[class_id] if labels_map else "{}".format(class_id)
            label_length = len(det_label)
            space_num_before = (len(classid_str) - label_length) // 2
            space_num_after = len(classid_str) - (space_num_before + label_length) + 2
            space_num_before_prob = (len(probability_str) - len(str(probs[class_id]))) // 2
            print("{}{}{}{}{:.7f}".format(' ' * space_num_before, det_label,
                                          ' ' * space_num_after, ' ' * space_num_before_prob,
                                          probs[class_id]))
        print("\n")

    log.info('This sample is an API example, for any performance measurements '
             'please use the dedicated benchmark_app tool')


if __name__ == '__main__':
    sys.exit(main() or 0)
