# -*- coding: utf-8 -*-
# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
import sys
from typing import Dict, List, Tuple

import numpy as np
from openvino.runtime import Output

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


def calculate_scale_factor(matrix: np.ndarray) -> float:
    """Get scale factor for quantization using utterance matrix."""
    # Max to find scale factor
    target_max = 16384
    max_val = np.max(matrix)
    if max_val == 0:
        return 1.0
    else:
        return target_max / max_val


def set_scale_factors(plugin_config: Dict[str, str], scale_factors: List[str], inputs: List[Output]):
    """Set a scale factor provided for each input."""
    for i in range(len(inputs)):
        log.info(f'For input {inputs[i].get_any_name()} using scale factor of {scale_factors[i]}')
        plugin_config[f'GNA_SCALE_FACTOR_{i}'] = scale_factors[i]


def get_input_layouts(layout_string: str, inputs: List[Output]) -> Dict[str, str]:
    if layout_string[0] == '[':
        return {_input.get_any_name(): layout_string[1:-1] for _input in inputs}
    else:
        sep = '],' if ',' in layout_string else ']'
        return dict([_input.split('[') for _input in layout_string[:-1].split(sep)])


def get_sorted_scale_factors(scale_factor_arg: Tuple[List[str], List[str]], inputs: List[Output]) -> List[str]:
    if scale_factor_arg[0]:
        res = [1 for _ in range(len(inputs))]
        input_names = [_input.get_any_name() for _input in inputs]

        for i in range(len(scale_factor_arg[0])):
            input_index = input_names.index(scale_factor_arg[0][i])
            res[input_index] = scale_factor_arg[1][i]

        return res

    else:
        return [scale_factor_arg[1][0] for _ in range(len(inputs))]
