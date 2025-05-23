# !/usr/bin/env python3

# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# pylint:disable=invalid-name,no-name-in-module,logging-format-interpolation,redefined-outer-name

""" Tool for running inference and storing results in npz files.
"""
import argparse
import logging as log
import os
import sys
from pathlib import Path

import numpy as np
from openvino import Core

log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)


def input_preparation(model):
    """
    Function to prepare reproducible from run to run input data
    :param model: OpenVINO Model object
    :return: Dict where keys are layers' names and values are numpy arrays with layers' shapes
    """

    feed_dict = {}
    for layer_name, layer_data in model.input_info.items():
        feed_dict.update({layer_name: np.ones(shape=layer_data.input_data.shape)})
    return feed_dict


def infer(ir_path, device):
    """
    Function to perform OV inference using python API "in place"
    :param ir_path: Path to XML file of IR
    :param device: Device name for inference
    :return: Dict containing out blob name and out data
    """

    core = Core()
    model = core.read_model(ir_path)
    compiled_model = core.compile_model(model, device)
    res = compiled_model(input_preparation(model))

    del model
    # It's important to delete compiled model first to avoid double free in plugin offloading.
    # Issue relates ony for hetero and Myriad plugins
    del compiled_model
    del core
    return res


def cli_parser():
    """
    Function for parsing arguments from command line.
    :return: ir path, device and output folder path variables.
    """
    parser = argparse.ArgumentParser(description='Arguments for python API inference')
    parser.add_argument('-m', dest='ir_path', required=True, help='Path to XML file of IR',  action="append")
    parser.add_argument('-d', dest='device', required=True, help='Target device to infer on')
    parser.add_argument('-r', dest='out_path', required=True, type=Path,
                        help='Dumps results to the output file')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true',
                        help='Increase output verbosity')
    args = parser.parse_args()
    ir_path = args.ir_path
    device = args.device
    out_path = args.out_path
    if args.verbose:
        log.getLogger().setLevel(log.DEBUG)
    return ir_path, device, out_path


if __name__ == "__main__":
    ir_path, device, out_path = cli_parser()

    for model in ir_path:
        result = infer(ir_path=model, device=device)

        np.savez(out_path / f"{Path(model).name}.npz", **result)

        log.info("Path for inference results: {}".format(out_path))
        log.debug("Inference results:")
        log.debug(result)
        log.debug("SUCCESS!")
