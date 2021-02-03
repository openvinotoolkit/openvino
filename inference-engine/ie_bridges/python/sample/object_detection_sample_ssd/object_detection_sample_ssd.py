#!/usr/bin/env python
"""
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
"""
from __future__ import print_function
import sys
import os
from argparse import ArgumentParser, SUPPRESS
import cv2
import numpy as np
import logging as log
from openvino.inference_engine import IECore

import ngraph as ng

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group("Options")
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml or .onnx file with a trained model.",
                      required=True, type=str)
    args.add_argument("-i", "--input", help="Required. Path to an image file.",
                      required=True, type=str)
    args.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. "
                           "Absolute path to a shared library with the kernels implementations.",
                      type=str, default=None)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; "
                           "CPU, GPU, FPGA or MYRIAD is acceptable. "
                           "Sample will look for a suitable plugin for device specified (CPU by default)",
                      default="CPU", type=str)
    args.add_argument("--labels", help="Optional. Labels mapping file", default=None, type=str)
    args.add_argument("-nt", "--number_top", help="Optional. Number of top results", default=10, type=int)

    return parser


def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    log.info("Loading Inference Engine")
    ie = IECore()

    # ---1. Read a model in OpenVINO Intermediate Representation (.xml and .bin files) or ONNX (.onnx file) format ---
    model = args.model
    log.info(f"Loading network:\n\t{model}")
    net = ie.read_network(model=model)
    # -----------------------------------------------------------------------------------------------------

    # ------------- 2. Load Plugin for inference engine and extensions library if specified --------------
    log.info("Device info:")
    versions = ie.get_versions(args.device)
    print(f"{' ' * 8}{args.device}")
    print(f"{' ' * 8}MKLDNNPlugin version ......... {versions[args.device].major}.{versions[args.device].minor}")
    print(f"{' ' * 8}Build ........... {versions[args.device].build_number}")

    if args.cpu_extension and "CPU" in args.device:
        ie.add_extension(args.cpu_extension, "CPU")
        log.info(f"CPU extension loaded: {args.cpu_extension}")
    # -----------------------------------------------------------------------------------------------------

    # --------------------------- 3. Read and preprocess input --------------------------------------------

    print("inputs number: " + str(len(net.input_info.keys())))
    assert len(net.input_info.keys()) == 1, 'Sample supports networks with one input'

    for input_key in net.input_info:
        print("input shape: " + str(net.input_info[input_key].input_data.shape))
        print("input key: " + input_key)
        if len(net.input_info[input_key].input_data.layout) == 4:
            n, c, h, w = net.input_info[input_key].input_data.shape

    images = np.ndarray(shape=(n, c, h, w))
    images_hw = []
    for i in range(n):
        image = cv2.imread(args.input)
        ih, iw = image.shape[:-1]
        images_hw.append((ih, iw))
        log.info("File was added: ")
        log.info(f"        {args.input}")
        if (ih, iw) != (h, w):
            log.warning(f"Image {args.input} is resized from {image.shape[:-1]} to {(h, w)}")
            image = cv2.resize(image, (w, h))
        image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        images[i] = image

    # -----------------------------------------------------------------------------------------------------

    # --------------------------- 4. Configure input & output ---------------------------------------------
    # --------------------------- Prepare input blobs -----------------------------------------------------
    log.info("Preparing input blobs")
    assert (len(net.input_info.keys()) == 1 or len(
        net.input_info.keys()) == 2), "Sample supports topologies only with 1 or 2 inputs"
    out_blob = next(iter(net.outputs))
    input_name, input_info_name = "", ""

    for input_key in net.input_info:
        if len(net.input_info[input_key].layout) == 4:
            input_name = input_key
            net.input_info[input_key].precision = 'U8'
        elif len(net.input_info[input_key].layout) == 2:
            input_info_name = input_key
            net.input_info[input_key].precision = 'FP32'
            if net.input_info[input_key].input_data.shape[1] != 3 and net.input_info[input_key].input_data.shape[1] != 6 or \
                net.input_info[input_key].input_data.shape[0] != 1:
                log.error('Invalid input info. Should be 3 or 6 values length.')

    data = {}
    data[input_name] = images

    if input_info_name != "":
        infos = np.ndarray(shape=(n, c), dtype=float)
        for i in range(n):
            infos[i, 0] = h
            infos[i, 1] = w
            infos[i, 2] = 1.0
        data[input_info_name] = infos

    # --------------------------- Prepare output blobs ----------------------------------------------------
    log.info('Preparing output blobs')

    func = ng.function_from_cnn(net)
    ops = func.get_ordered_ops()
    output_name, output_info = "", net.outputs[next(iter(net.outputs.keys()))]
    output_ops = {op.friendly_name : op for op in ops \
                  if op.friendly_name in net.outputs and op.get_type_name() == "DetectionOutput"}
    if len(output_ops) != 0:
        output_name, output_info = output_ops.popitem()

    if output_name == "":
        log.error("Can't find a DetectionOutput layer in the topology")

    output_dims = output_info.shape
    if len(output_dims) != 4:
        log.error("Incorrect output dimensions for SSD model")
    max_proposal_count, object_size = output_dims[2], output_dims[3]

    if object_size != 7:
        log.error("Output item should have 7 as a last dimension")

    output_info.precision = "FP32"
    # -----------------------------------------------------------------------------------------------------

    # --------------------------- Performing inference ----------------------------------------------------
    log.info("Loading model to the device")
    exec_net = ie.load_network(network=net, device_name=args.device)
    log.info("Creating infer request and starting inference")
    res = exec_net.infer(inputs=data)
    # -----------------------------------------------------------------------------------------------------

    # --------------------------- Read and postprocess output ---------------------------------------------
    log.info("Processing output blobs")
    res = res[out_blob]
    boxes, classes = {}, {}
    data = res[0][0]
    for number, proposal in enumerate(data):
        if proposal[2] > 0:
            imid = np.int(proposal[0])
            ih, iw = images_hw[imid]
            label = np.int(proposal[1])
            confidence = proposal[2]
            xmin = np.int(iw * proposal[3])
            ymin = np.int(ih * proposal[4])
            xmax = np.int(iw * proposal[5])
            ymax = np.int(ih * proposal[6])
            print(f"[{number},{label}] element, prob = {confidence:.6f}    ({xmin},{ymin})-({xmax},{ymax}) "
                  f"batch id : {imid}", end="")
            if proposal[2] > 0.5:
                print(" WILL BE PRINTED!")
                if not imid in boxes.keys():
                    boxes[imid] = []
                boxes[imid].append([xmin, ymin, xmax, ymax])
                if not imid in classes.keys():
                    classes[imid] = []
                classes[imid].append(label)
            else:
                print()

    for imid in classes:
        tmp_image = cv2.imread(args.input)
        for box in boxes[imid]:
            cv2.rectangle(tmp_image, (box[0], box[1]), (box[2], box[3]), (232, 35, 244), 2)
        cv2.imwrite("out.bmp", tmp_image)
        log.info("Image out.bmp created!")
    # -----------------------------------------------------------------------------------------------------

    log.info("Execution successful\n")
    log.info(
        "This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool")


if __name__ == '__main__':
    sys.exit(main() or 0)
