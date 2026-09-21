# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# !/usr/bin/env python3
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


def input_preparation(compiled_model):
    """
    Function to prepare reproducible from run to run input data
    :param compiled_model: OpenVINO CompiledModel object
    :return: Dict where keys are input ports and values are numpy arrays with input shapes
    """

    feed_dict = {}
    # Key by the ConstOutput port (not any_name): a port isn't guaranteed to have tensor
    # names, and Model.inputs' plain Output isn't a valid key for compiled_model().
    for model_input in compiled_model.inputs:
        feed_dict[model_input] = np.ones(shape=list(model_input.shape))
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
    res = compiled_model(input_preparation(compiled_model))

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
        # OVDict keys are ports, not strings; np.savez needs string keyword keys.
        named_result = {(port.any_name if port.get_names() else f"output_{i}"): value for i, (port, value) in enumerate(result.items())}

        np.savez(out_path / f"{Path(model).name}.npz", **named_result)

        log.info("Path for inference results: {}".format(out_path))
        log.debug("Inference results:")
        log.debug(result)
        log.debug("SUCCESS!")
