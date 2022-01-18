# -*- coding: utf-8 -*-
# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging as log
import re
import sys
from typing import List, Tuple

import numpy as np

# Operating Frequency for GNA HW devices for Core and Atom architecture
GNA_CORE_FREQUENCY = 400
GNA_ATOM_FREQUENCY = 200

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)


def compare_with_reference(result: np.ndarray, reference: np.ndarray):
    error_matrix = np.absolute(result - reference)

    max_error = np.max(error_matrix)
    sum_error = np.sum(error_matrix)
    avg_error = sum_error / error_matrix.size
    sum_square_error = np.sum(np.square(error_matrix))
    avg_rms_error = np.sqrt(sum_square_error / error_matrix.size)
    stdev_error = np.sqrt(sum_square_error / error_matrix.size - avg_error * avg_error)

    log.info(f'max error: {max_error:.7f}')
    log.info(f'avg error: {avg_error:.7f}')
    log.info(f'avg rms error: {avg_rms_error:.7f}')
    log.info(f'stdev error: {stdev_error:.7f}')


def get_scale_factor(matrix: np.ndarray) -> float:
    """Get scale factor for quantization using utterance matrix"""
    # Max to find scale factor
    target_max = 16384
    max_val = np.max(matrix)
    if max_val == 0:
        return 1.0
    else:
        return target_max / max_val


def set_scale_factors(plugin_config: dict, scale_factors: list):
    """Set a scale factor provided for each input"""
    for i, scale_factor in enumerate(scale_factors):
        log.info(f'For input {i} using scale factor of {scale_factor:.7f}')
        plugin_config[f'GNA_SCALE_FACTOR_{i}'] = str(scale_factor)


def parse_scale_factors(args: argparse.Namespace) -> list:
    """Get a list of scale factors for input files"""
    input_files = re.split(', |,', args.input)
    scale_factors = re.split(', |,', str(args.scale_factor))
    scale_factors = list(map(float, scale_factors))

    if len(input_files) != len(scale_factors):
        log.error(f'Incorrect command line for multiple inputs: {len(scale_factors)} scale factors provided for '
                  f'{len(input_files)} input files.')
        sys.exit(-7)

    for i, scale_factor in enumerate(scale_factors):
        if float(scale_factor) < 0:
            log.error(f'Scale factor for input #{i} (counting from zero) is out of range (must be positive).')
            sys.exit(-8)

    return scale_factors


def parse_outputs_from_args(args: argparse.Namespace) -> Tuple[List[str], List[int]]:
    """Get a list of outputs specified in the args"""
    name_and_port = [output.split(':') for output in re.split(', |,', args.output_layers)]
    try:
        return [name for name, _ in name_and_port], [int(port) for _, port in name_and_port]
    except ValueError:
        log.error('Incorrect value for -oname/--output_layers option, please specify a port for each output layer.')
        sys.exit(-4)
