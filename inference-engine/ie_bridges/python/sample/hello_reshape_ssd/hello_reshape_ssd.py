#!/usr/bin/env python
'''
 Copyright (c) 2018 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
'''
from __future__ import print_function
from argparse import ArgumentParser, SUPPRESS
import logging as log
import os
import sys

import cv2
import ngraph as ng
import numpy as np
from openvino.inference_engine import IECore

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument('-m', '--model', help='Required. Path to an .xml or .onnx file with a trained model.',
                      required=True, type=str)
    args.add_argument('-i', '--input', help='Required. Path to an image file.',
                      required=True, type=str)
    args.add_argument('-d', '--device',
                      help='Optional. Specify the target device to infer on; '
                           'CPU, GPU, FPGA or MYRIAD is acceptable. '
                           'Sample will look for a suitable plugin for device specified (CPU by default)',
                      default='CPU', type=str)
    return parser


def main():
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    log.info('Loading Inference Engine')
    ie = IECore()

    # ---1. Read a model in OpenVINO Intermediate Representation (.xml and .bin files) or ONNX (.onnx file) format ---
    model = args.model
    log.info(f'Loading network:')
    log.info(f'    {model}')
    net = ie.read_network(model=model)
    # -----------------------------------------------------------------------------------------------------

    # ------------- 2. Load Plugin for inference engine and extensions library if specified --------------
    log.info('Device info:')
    versions = ie.get_versions(args.device)
    log.info(f'        {args.device}')
    log.info(f'        MKLDNNPlugin version ......... {versions[args.device].major}.{versions[args.device].minor}')
    log.info(f'        Build ........... {versions[args.device].build_number}')
    # -----------------------------------------------------------------------------------------------------

    # --------------------------- 3. Read and preprocess input --------------------------------------------

    log.info(f'Inputs number: {len(net.input_info.keys())}')
    assert len(net.input_info.keys()) == 1, 'Sample supports clean SSD network with one input'
    assert len(net.outputs.keys()) == 1, 'Sample supports clean SSD network with one output'
    input_name = list(net.input_info.keys())[0]
    input_info = net.input_info[input_name]
    supported_input_dims = 4  # Supported input layout - NHWC

    log.info(f'    Input name: {input_name}')
    log.info(f'    Input shape: {str(input_info.input_data.shape)}')
    if len(input_info.input_data.layout) == supported_input_dims:
        n, c, h, w = input_info.input_data.shape
        assert n == 1, 'Sample supports topologies with one input image only'
    else:
        raise AssertionError('Sample supports input with NHWC shape only')
    
    image = cv2.imread(args.input)
    h_new, w_new = image.shape[:-1]
    images = np.ndarray(shape=(n, c, h_new, w_new))
    log.info('File was added: ')
    log.info(f'    {args.input}')
    image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    images[0] = image
    
    log.info('Reshaping the network to the height and width of the input image')
    net.reshape({input_name: [n, c, h_new, w_new]})
    log.info(f'Input shape after reshape: {str(net.input_info["data"].input_data.shape)}')

    # -----------------------------------------------------------------------------------------------------

    # --------------------------- 4. Configure input & output ---------------------------------------------
    # --------------------------- Prepare input blobs -----------------------------------------------------
    log.info('Preparing input blobs')

    if len(input_info.layout) == supported_input_dims:
        input_info.precision = 'U8'
    
    data = {}
    data[input_name] = images

    # --------------------------- Prepare output blobs ----------------------------------------------------
    log.info('Preparing output blobs')

    func = ng.function_from_cnn(net)
    ops = func.get_ordered_ops()
    output_name, output_info = '', net.outputs[next(iter(net.outputs.keys()))]
    output_ops = {op.friendly_name : op for op in ops \
                  if op.friendly_name in net.outputs and op.get_type_name() == 'DetectionOutput'}
    if len(output_ops) == 1:
        output_name, output_info = output_ops.popitem()

    assert output_name != '', 'Can''t find a DetectionOutput layer in the topology'

    output_dims = output_info.shape
    assert output_dims != 4, 'Incorrect output dimensions for SSD model'
    assert output_dims[3] == 7, 'Output item should have 7 as a last dimension'

    output_info.precision = 'FP32'
    # -----------------------------------------------------------------------------------------------------

    # --------------------------- Performing inference ----------------------------------------------------
    log.info('Loading model to the device')
    exec_net = ie.load_network(network=net, device_name=args.device)
    log.info('Creating infer request and starting inference')
    exec_result = exec_net.infer(inputs=data)
    # -----------------------------------------------------------------------------------------------------

    # --------------------------- Read and postprocess output ---------------------------------------------
    log.info('Processing output blobs')
    result = exec_result[output_name]
    boxes = {}
    detections = result[0][0]  # [0][0] - location of detections in result blob
    for number, proposal in enumerate(detections):
        imid, label, confidence, coords = np.int(proposal[0]), np.int(proposal[1]), proposal[2], proposal[3:]
        if confidence > 0.5:
            # correcting coordinates to actual image resolution
            xmin, ymin, xmax, ymax = w_new * coords[0], h_new * coords[1], w_new * coords[2], h_new * coords[3]

            log.info(f'    [{number},{label}] element, prob = {confidence:.6f}, '
                f'bbox = ({xmin:.3f},{ymin:.3f})-({xmax:.3f},{ymax:.3f}), batch id = {imid}')
            if not imid in boxes.keys():
                boxes[imid] = []
            boxes[imid].append([xmin, ymin, xmax, ymax])

    imid = 0  # as sample supports one input image only, imid in results will always be 0

    tmp_image = cv2.imread(args.input)
    for box in boxes[imid]:
        # drawing bounding boxes on the output image
        cv2.rectangle(
            tmp_image, 
            (np.int(box[0]), np.int(box[1])), (np.int(box[2]), np.int(box[3])), 
            color=(232, 35, 244), thickness=2)
    cv2.imwrite('out.bmp', tmp_image)
    log.info('Image out.bmp created!')
    # -----------------------------------------------------------------------------------------------------

    log.info('Execution successful\n')
    log.info(
        'This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool')


if __name__ == '__main__':
    sys.exit(main() or 0)
