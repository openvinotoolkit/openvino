#!/usr/bin/env python3
import sys
from argparse import ArgumentParser, SUPPRESS
import cv2
import numpy as np
import logging as log
from openvino.inference_engine import IECore, IENetwork
import ngraph
from ngraph.impl import Function
from functools import reduce
import pathlib


def build_argparser() -> ArgumentParser:
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument('-i', '--input', help='Required. Path to a folder with images or path to an image files',
                      required=True,
                      type=str, nargs="+")
    args.add_argument('-m', '--model', help='Required. Path to file where weights for the network are located')
    args.add_argument('-l', '--cpu_extension',
                      help='Optional. Required for CPU custom layers. '
                           'MKLDNN (CPU)-targeted custom layers. Absolute path to a shared library with the'
                           ' kernels implementations.', type=str, default=None)
    args.add_argument('-d', '--device',
                      help='Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL, MYRIAD or HETERO: is '
                           'acceptable. The sample will look for a suitable plugin for device specified. Default '
                           'value is CPU',
                      default='CPU', type=str)
    args.add_argument('--labels', help='Optional. Path to a labels mapping file', default=None, type=str)
    # args.add_argument('-nt', '--number_top', help='Optional. Number of top results', default=10, type=int)

    return parser


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
    add_1_kernel = ngraph.constant(weights[weights_offset:weights_offset + add_1_kernel_length].reshape(add_1_kernel_shape))
    weights_offset += add_1_kernel_length
    add_1_node = ngraph.add(conv_1_node, add_1_kernel)

    # maxpool 1
    maxpool_1_node = ngraph.max_pool(add_1_node, [2, 2], padding_begin, padding_end, [2, 2], 'ceil', None)

    # convolution 2
    conv_2_kernel_shape, conv_2_kernel_length = shape_and_length([50, 20, 5, 5])
    conv_2_kernel = ngraph.constant(weights[weights_offset:weights_offset + conv_2_kernel_length].reshape(conv_2_kernel_shape))
    weights_offset += conv_2_kernel_length
    conv_2_node = ngraph.convolution(maxpool_1_node, conv_2_kernel, [1, 1], padding_begin, padding_end, [1, 1])

    # add 2
    add_2_kernel_shape, add_2_kernel_length = shape_and_length([1, 50, 1, 1])
    add_2_kernel = ngraph.constant(weights[weights_offset:weights_offset + add_2_kernel_length].reshape(add_2_kernel_shape))
    weights_offset += add_2_kernel_length
    add_2_node = ngraph.add(conv_2_node, add_2_kernel)

    # maxpool 2
    maxpool_2_node = ngraph.max_pool(add_2_node, [2, 2], padding_begin, padding_end, [2, 2], 'ceil', None)

    # reshape 1
    reshape_1_dims, reshape_1_length = shape_and_length([2])
    dtype_weights = np.frombuffer(weights[weights_offset:weights_offset + 2*reshape_1_length], dtype=np.int64)  # workaround to get int64 weights from float32 ndarray w/o unnecessary copying
    reshape_1_kernel = ngraph.constant(dtype_weights)
    weights_offset += 2*reshape_1_length
    reshape_1_node = ngraph.reshape(maxpool_2_node, reshape_1_kernel, True)

    # matmul 1
    matmul_1_kernel_shape, matmul_1_kernel_length = shape_and_length([500, 800])
    matmul_1_kernel = ngraph.constant(weights[weights_offset:weights_offset + matmul_1_kernel_length].reshape(matmul_1_kernel_shape))
    weights_offset += matmul_1_kernel_length
    matmul_1_node = ngraph.matmul(reshape_1_node, matmul_1_kernel, False, True)

    # add 3
    add_3_kernel_shape, add_3_kernel_length = shape_and_length([1, 500])
    add_3_kernel = ngraph.constant(weights[weights_offset:weights_offset + add_3_kernel_length].reshape(add_3_kernel_shape))
    weights_offset += add_3_kernel_length
    add_3_node = ngraph.add(matmul_1_node, add_3_kernel)

    # ReLU
    relu_node = ngraph.relu(add_3_node)

    # reshape 2
    reshape_2_dims, reshape_2_length = shape_and_length([2])
    reshape_2_kernel = ngraph.constant(dtype_weights)
    reshape_2_node = ngraph.reshape(relu_node, reshape_2_kernel, True)

    # matmul 2
    matmul_2_kernel_shape, matmul_2_kernel_length = shape_and_length([10, 500])
    matmul_2_kernel = ngraph.constant(weights[weights_offset:weights_offset + matmul_2_kernel_length].reshape(matmul_2_kernel_shape))
    weights_offset += matmul_2_kernel_length
    matmul_2_node = ngraph.matmul(reshape_2_node, matmul_2_kernel, False, True)

    # add 4
    add_4_kernel_shape, add_4_kernel_length = shape_and_length([1, 10])
    add_4_kernel = ngraph.constant(weights[weights_offset:weights_offset + add_4_kernel_length].reshape(add_4_kernel_shape))
    weights_offset += add_4_kernel_length
    add_4_node = ngraph.add(matmul_2_node, add_4_kernel)

    # softmax
    softmax_node = ngraph.softmax(add_4_node, 1)  # 1 is softmax application axis

    # result
    result_node = ngraph.result(softmax_node)

    # nGraph function
    function = Function(result_node, [param_node], 'lenet')

    return function


def main():
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    log.info('Initializing Inference Engine')
    ie = IECore()

    # Read and pre-process input images
    input = np.ndarray(shape=(64, 1, 28, 28))
    path = pathlib.Path(args.input[0])
    if path.is_dir():
        for index, file in enumerate(sorted(path.iterdir())):
            image = cv2.imread(str(file))
            input[index][0] = image.transpose(2, 0, 1)[0]  # works for MNIST saved via cv2.imsave()
    elif path.is_file() and path.suffixes[-1] == '.npy':
        input = np.load(path, allow_pickle=False)
        if input.shape != (64, 1, 28, 28):
            log.error('If .npy file is passed as input, its data has to have a shape (64, 1, 28, 28)')
    else:
        log.error('Please, pass directory or file containing numpy.ndarray with correct shape')
        exit(-1)

    ngraph_function = create_ngraph_function(args)
    net = IENetwork(Function.to_capsule(ngraph_function))  # using deprecated IENetwork constructor

    log.info('Loading model to the device')
    exec_net = ie.load_network(network=net, device_name=args.device.upper())

    log.info('Creating infer request and starting inference')
    data = {next(iter(net.input_info)): input}

    log.info('Processing results')
    res = exec_net.infer(inputs=data)
    res = res[next(iter(net.outputs))]
    if path.is_dir():
        for index, file in enumerate(sorted(path.iterdir())):
            log.info(f'File: {file.name}, class id: {res[index].argmax()}')
    elif path.is_file():
        for index in range(0, input.shape[0]):
            log.info(f'Image No.{index}, class id: {res[index].argmax()}')

    log.info('This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool')


if __name__ == '__main__':
    sys.exit(main() or 0)
