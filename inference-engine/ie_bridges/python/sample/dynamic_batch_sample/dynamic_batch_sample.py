#!/usr/bin/env python
"""
 Copyright (C) 2018-2019 Intel Corporation

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
from time import time
from openvino.inference_engine import IENetwork, IEPlugin


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.", required=True,
                      type=str)
    args.add_argument("-i", "--input", help="Required. Path to a folder with images or path to an image files",
                      required=True, type=str, nargs="+")
    args.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. Absolute path to a shared library with the "
                           "kernels implementations.", type=str, default=None)
    args.add_argument("-pp", "--plugin_dir", help="Optional. Path to a plugin folder", type=str, default=None)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is "
                           "acceptable. The sample will look for a suitable plugin for device specified. "
                           "Default value is CPU", default="CPU", type=str)
    args.add_argument("--labels", help="Optional. Path to labels mapping file", default=None, type=str)
    args.add_argument("-mb", "--max_batch", help="Optional. Set maximum batch size for the network", default=20,
                      type=int)
    args.add_argument("-ni", "--number_iter", help="Optional. Number of inference iterations", default=1, type=int)
    args.add_argument("-pc", "--perf_counts", help="Optional. Report performance counters", default=False,
                      action="store_true")

    return parser


def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"

    # Plugin initialization for specified device and load extensions library if specified
    plugin = IEPlugin(device=args.device, plugin_dirs=args.plugin_dir)

    # Configure plugin to support dynamic batch size
    plugin.set_config({"DYN_BATCH_ENABLED": "YES"})

    # Load cpu_extensions library if specified
    if args.cpu_extension and 'CPU' in args.device:
        plugin.add_cpu_extension(args.cpu_extension)

    # Read IR
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = IENetwork(model=model_xml, weights=model_bin)

    # Check for unsupported layers if the device is 'CPU'
    if plugin.device == "CPU":
        unsupported_layers = [layer for layer in net.layers if layer not in plugin.get_supported_layers(net)]
        if len(unsupported_layers) != 0:
            log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(plugin.device, ', '.join(unsupported_layers)))
            log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)

    assert len(net.inputs.keys()) == 1, "Sample supports only single input topologies"

    log.info("Preparing input blobs")
    input_blob = next(iter(net.inputs))

    # Set max batch size
    inputs_count = len(args.input)
    if args.max_batch < inputs_count:
        log.warning("Defined max_batch size {} less than input images count {}."
                    "\n\t\t\tInput images count will be used as max batch size".format(args.max_batch, inputs_count))
    net.batch_size = max(args.max_batch, inputs_count)

    # Create numpy array for the max_batch size images
    n, c, h, w = net.inputs[input_blob].shape
    images = np.zeros(shape=(n, c, h, w))

    # Read and pre-process input images
    for i in range(inputs_count):
        image = cv2.imread(args.input[i])
        if image.shape[:-1] != (h, w):
            log.warning("Image {} is resized from {} to {}".format(args.input[i], image.shape[:-1], (h, w)))
            image = cv2.resize(image, (w, h))
        image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        images[i] = image
    log.info("Batch size is {}".format(n))

    # Loading model to the plugin
    log.info("Loading model to the plugin")
    exec_net = plugin.load(network=net)

    def infer():
        for i in range(args.number_iter):
            t0 = time()
            exec_net.infer(inputs={input_blob: images})
            infer_time.append((time() - t0) * 1000)
        log.info("Average running time of one iteration: {} ms".format(np.average(np.asarray(infer_time))))
        if args.perf_counts:
            perf_counts = exec_net.requests[0].get_perf_counts()
            log.info("Performance counters:")
            print("{:<70} {:<15} {:<15} {:<15} {:<10}".format('name', 'layer_type', 'exet_type', 'status',
                                                              'real_time, us'))
            for layer, stats in perf_counts.items():
                print("{:<70} {:<15} {:<15} {:<15} {:<10}".format(layer, stats['layer_type'], stats['exec_type'],
                                                                  stats['status'], stats['real_time']))

    # Start sync inference with full batch size
    log.info(
        "Starting inference with full batch {} ({} iterations)".format(n, args.number_iter)
    )
    infer_time = []
    infer()

    # Set batch size dynamically for the infer request and start sync inference
    infer_time = []
    exec_net.requests[0].set_batch(inputs_count)
    log.info("Starting inference with dynamically defined batch {} for the 2nd infer request ({} iterations)".format(
        inputs_count, args.number_iter))
    infer()


if __name__ == '__main__':
    sys.exit(main() or 0)
