#!/usr/bin/env python
"""
 Copyright (C) 2018-2021 Intel Corporation

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
from ngraph.impl import Function, Dimension, PartialShape
from functools import reduce
import struct as st


def build_argparser() -> ArgumentParser:
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument('-i', '--input', help='Required. Path to a folder with images or path to an image files',
                      required=False, type=str, nargs="+")
    args.add_argument('-m', '--model', help='Required. Path to file where weights for the network are located',
                      required=False)
    args.add_argument('-d', '--device',
                      help='Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL, MYRIAD or HETERO: '
                           'is acceptable. The sample will look for a suitable plugin for device specified. Default '
                           'value is CPU',
                      default='CPU', type=str)
    args.add_argument('--labels', help='Optional. Path to a labels mapping file', default=None, type=str)
    args.add_argument('-nt', '--number_top', help='Optional. Number of top results', default=1, type=int)

    return parser


def list_input_images(input_dirs: list):
    return []


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

    # input
    input_shape = PartialShape([Dimension(64), Dimension(), Dimension(28), Dimension(28)])
    param_node = ngraph.parameter(input_shape, np.float32, 'Parameter')


    multiply = ngraph.multiply(param_node, ngraph.constant(np.float32(2.0)))


    # result
    result_node = ngraph.result(multiply)

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

    net.reshapePartial({'Parameter_0': [-1, 2, [2, 5], (0, 28)]})

    for output in net.outputs:
        print(output, net.outputs[output])

    assert len(net.input_info.keys()) == 1, "Sample supports only single input topologies"
    assert len(net.outputs) == 1, "Sample supports only single output topologies"

    log.info("Preparing input blobs")
    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))
    ie = IECore()

    log.info('Loading model to the device')
    exec_net = ie.load_network(network=net, device_name=args.device.upper())

    # Start sync inference
    log.info('Creating infer request and starting inference')
    res = exec_net.infer(inputs={input_blob: np.zeros(shape=[64, 2, 2, 28])})

    # Processing results
    log.info("Processing output blob")
    res = res[out_blob]
    log.info(f"Top {args.number_top} results: ")

    print("res.shape = ", res.shape)

    # Start sync inference
    log.info('Creating infer request and starting inference')
    res = exec_net.infer(inputs={input_blob: np.ones(shape=[1, 2, 2, 1])})

    # Processing results
    log.info("Processing output blob")
    res = res[out_blob]
    log.info(f"Top {args.number_top} results: ")

    print("res.shape = ", res.shape)
    print(res)

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
        print(f"Image {input_images[i]}\n")
        print(classid_str, probability_str)
        print(f"{'-' * len(classid_str)} {'-' * len(probability_str)}")
        for class_id in top_ind:
            det_label = labels_map[class_id] if labels_map else f"{class_id}"
            label_length = len(det_label)
            space_num_before = (len(classid_str) - label_length) // 2
            space_num_after = len(classid_str) - (space_num_before + label_length) + 2
            print(f"{' ' * space_num_before}{det_label}"
                  f"{' ' * space_num_after}{probs[class_id]:.7f}")
        print("\n")

    log.info('This sample is an API example, for any performance measurements '
             'please use the dedicated benchmark_app tool')


if __name__ == '__main__':
    sys.exit(main() or 0)
