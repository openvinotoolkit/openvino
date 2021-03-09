# !/usr/bin/env python3
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# pylint:disable=invalid-name,no-name-in-module,logging-format-interpolation,redefined-outer-name

""" Tool for running inference and storing results in npz files.
"""
import argparse
import logging as log
import sys
import os
import numpy as np
from openvino.inference_engine import IECore

log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)


def input_preparation(net):
    """
    Function to prepare reproducible from run to run input data
    :param net: IENetwork object
    :return: Dict where keys are layers' names and values are numpy arrays with layers' shapes
    """

    feed_dict = {}
    for layer_name, layer_data in net.inputs.items():
        feed_dict.update({layer_name: np.ones(shape=layer_data.shape)})
    return feed_dict


def infer(ir_path, device):
    """
    Function to perform IE inference using python API "in place"
    :param ir_path: Path to XML file of IR
    :param device: Device name for inference
    :return: Dict containing out blob name and out data
    """

    bin_path = os.path.splitext(ir_path)[0] + '.bin'
    ie = IECore()
    net = ie.read_network(model=ir_path, weights=bin_path)
    exec_net = ie.load_network(net, device)
    res = exec_net.infer(inputs=input_preparation(net))

    del net
    # It's important to delete executable network first to avoid double free in plugin offloading.
    # Issue relates ony for hetero and Myriad plugins
    del exec_net
    del ie
    return res


def cli_parser():
    """
    Function for parsing arguments from command line.
    :return: ir path, device and output folder path variables.
    """
    parser = argparse.ArgumentParser(description='Arguments for python API inference')
    parser.add_argument('-m', dest='ir_path', required=True, help='Path to XML file of IR')
    parser.add_argument('-d', dest='device', required=True, help='Target device to infer on')
    parser.add_argument('-r', dest='out_path', required=True,
                        help='Dumps results to the output file')
    args = parser.parse_args()
    ir_path = args.ir_path
    device = args.device
    out_path = args.out_path
    return ir_path, device, out_path


if __name__ == "__main__":
    ir_path, device, out_path = cli_parser()
    results = infer(ir_path=ir_path, device=device)
    np.savez(out_path, **results)
    log.info("Path for inference results: {}".format(out_path))
    log.info("Inference results:")
    log.info(results)
    log.info("SUCCESS!")
