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
from openvino.inference_engine import IENetwork, IEPlugin


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.", required=True,
                      type=str)
    args.add_argument("-i", "--input", help="Required. Path to a folder with images or path to an image files",
                      required=True, type=str)
    args.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. "
                           "Absolute path to a shared library with the kernels implementations.",
                      type=str, default=None)
    args.add_argument("-pp", "--plugin_dir", help="Optional. Path to a plugin folder", type=str, default=None)
    args.add_argument("-d", "--device",
                      help="Optional. Specify HETERO plugin configuration; for example, HETERO:FPGA,CPU",
                      default="HETERO:CPU,GPU", type=str)
    args.add_argument("-nt", "--number_top", help="Optional. Number of top results", default=10, type=int)

    return parser


def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    assert args.device.split(':')[0] == "HETERO", "This sample supports only Hetero Plugin. " \
                                                  "Please specify correct device, e.g. HETERO:FPGA,CPU"
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"

    # Plugin initialization for specified device and load extensions library if specified
    plugin = IEPlugin(device=args.device, plugin_dirs=args.plugin_dir)
    if args.cpu_extension and 'CPU' in args.device:
        plugin.add_cpu_extension(args.cpu_extension)
    # Read IR
    net = IENetwork(model=model_xml, weights=model_bin)

    if plugin.device == "CPU":
        supported_layers = plugin.get_supported_layers(net)
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(plugin.device, ', '.join(not_supported_layers)))
            log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)
    net_ops = set([l.type for l in net.layers.values()])
    if not any([op == "Convolution" for op in net_ops]):
        log.warning("Specified IR doesn't contain any Convolution operations for which affinity going to be set.\n"
                    "Try to use another topology to make the affinity setting result more visible.")

    # Configure the plugin to initialize default affinity for network in set_initial_affinity() function.
    plugin.set_config({"TARGET_FALLBACK": args.device.split(':')[1]})
    # Enable graph visualization
    plugin.set_config({"HETERO_DUMP_GRAPH_DOT": "YES"})
    plugin.set_initial_affinity(net)

    for l in net.layers.values():
        if l.type == "Convolution":
            l.affinity = "GPU"

    assert len(net.inputs.keys()) == 1, "Sample supports only single input topologies"
    assert len(net.outputs) == 1, "Sample supports only single output topologies"
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    # Read and pre-process input image
    n, c, h, w = net.inputs[input_blob].shape
    image = cv2.imread(args.input)
    image = cv2.resize(image, (w, h))
    image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    image = image.reshape((n, c, h, w))
    # Load network to the plugin
    exec_net = plugin.load(network=net)
    del net
    # Start sync inference
    res = exec_net.infer(inputs={input_blob: image})
    top_ind = np.argsort(res[out_blob], axis=1)[0, -args.number_top:][::-1]
    for i in top_ind:
        log.info("%f #%d" % (res[out_blob][0, i], i))
    del exec_net
    del plugin
    cwd = os.getcwd()
    log.info(
        "Graphs representing default and resulting affinities dumped to {} and {} files respectively"
            .format(os.path.join(cwd, 'hetero_affinity.dot'), os.path.join(cwd, 'hetero_subgraphs.dot'))
    )


if __name__ == '__main__':
    sys.exit(main() or 0)
