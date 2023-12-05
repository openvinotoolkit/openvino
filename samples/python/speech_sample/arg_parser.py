# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import argparse
import re
from typing import List, Tuple, Union


def build_arg_parser() -> argparse.ArgumentParser:
    """Create and return argument parser."""
    parser = argparse.ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    model = parser.add_mutually_exclusive_group(required=True)

    model.add_argument('-m', '--model', type=str,
                       help='Path to an .xml file with a trained model (required if -rg is missing).')
    model.add_argument('-rg', '--import_gna_model', type=str,
                       help='Read GNA model from file using path/filename provided (required if -m is missing).')

    args.add_argument('-h', '--help', action='help', help='Show this help message and exit.')
    args.add_argument('-i', '--input', required=True, type=str,
                      help='Required. Path(s) to input file(s). '
                      'Usage for a single file/layer: <input_file.ark> or <input_file.npz>. '
                      'Example of usage for several files/layers: <layer1>:<port_num1>=<input_file1.ark>,<layer2>:<port_num2>=<input_file2.ark>.')
    args.add_argument('-o', '--output', type=str,
                      help='Optional. Output file name(s) to save scores (inference results). '
                      'Usage for a single file/layer: <output_file.ark> or <output_file.npz>. '
                      'Example of usage for several files/layers: <layer1>:<port_num1>=<output_file1.ark>,<layer2>:<port_num2>=<output_file2.ark>.')
    args.add_argument('-r', '--reference', type=str,
                      help='Optional. Read reference score file(s) and compare inference results with reference scores. '
                      'Usage for a single file/layer: <reference_file.ark> or <reference_file.npz>. '
                      'Example of usage for several files/layers: <layer1>:<port_num1>=<reference_file1.ark>,<layer2>:<port_num2>=<reference_file2.ark>.')
    args.add_argument('-d', '--device', default='CPU', type=str,
                      help='Optional. Specify a target device to infer on. '
                      'CPU, GPU, NPU, GNA_AUTO, GNA_HW, GNA_SW_FP32, GNA_SW_EXACT and HETERO with combination of GNA'
                      ' as the primary device and CPU as a secondary (e.g. HETERO:GNA,CPU) are supported. '
                      'The sample will look for a suitable plugin for device specified. Default value is CPU.')
    args.add_argument('-bs', '--batch_size', type=int, choices=range(1, 9), metavar='[1-8]',
                      help='Optional. Batch size 1-8.')
    args.add_argument('-layout', type=str,
                      help='Optional. Custom layout in format: "input0[value0],input1[value1]" or "[value]" (applied to all inputs)')
    args.add_argument('-qb', '--quantization_bits', default=16, type=int, choices=(8, 16), metavar='[8, 16]',
                      help='Optional. Weight resolution in bits for GNA quantization: 8 or 16 (default 16).')
    args.add_argument('-sf', '--scale_factor', type=str,
                      help='Optional. User-specified input scale factor for GNA quantization. '
                      'If the model contains multiple inputs, provide scale factors by separating them with commas. '
                      'For example: <layer1>:<sf1>,<layer2>:<sf2> or just <sf> to be applied to all inputs.')
    args.add_argument('-wg', '--export_gna_model', type=str,
                      help='Optional. Write GNA model to file using path/filename provided.')
    args.add_argument('-we', '--export_embedded_gna_model', type=str,
                      help='Optional. Write GNA embedded model to file using path/filename provided.')
    args.add_argument('-we_gen', '--embedded_gna_configuration', default='GNA1', type=str, metavar='[GNA1, GNA3]',
                      help='Optional. GNA generation configuration string for embedded export. '
                      'Can be GNA1 (default) or GNA3.')
    args.add_argument('--exec_target', default='', type=str, choices=('GNA_TARGET_2_0', 'GNA_TARGET_3_0'),
                      metavar='[GNA_TARGET_2_0, GNA_TARGET_3_0]',
                      help='Optional. Specify GNA execution target generation. '
                      'By default, generation corresponds to the GNA HW available in the system '
                      'or the latest fully supported generation by the software. '
                      "See the GNA Plugin's GNA_EXEC_TARGET config option description.")
    args.add_argument('-pc', '--performance_counter', action='store_true',
                      help='Optional. Enables performance report (specify -a to ensure arch accurate results).')
    args.add_argument('-a', '--arch', default='CORE', type=str.upper, choices=('CORE', 'ATOM'), metavar='[CORE, ATOM]',
                      help='Optional. Specify architecture. CORE, ATOM with the combination of -pc.')
    args.add_argument('-cw_l', '--context_window_left', type=int, default=0,
                      help='Optional. Number of frames for left context windows (default is 0). '
                      'Works only with context window models. '
                      'If you use the cw_l or cw_r flag, then batch size argument is ignored.')
    args.add_argument('-cw_r', '--context_window_right', type=int, default=0,
                      help='Optional. Number of frames for right context windows (default is 0). '
                      'Works only with context window models. '
                      'If you use the cw_l or cw_r flag, then batch size argument is ignored.')
    args.add_argument('-pwl_me', type=float, default=1.0,
                      help='Optional. The maximum percent of error for PWL function. '
                      'The value must be in <0, 100> range. The default value is 1.0.')

    return parser


def parse_arg_with_names(arg_string: Union[str, None], separator: str = '=') -> Tuple[List[str], List[str]]:
    keys = []
    values = []

    if isinstance(arg_string, str):
        for parameter in re.split(', |,', arg_string):
            if separator in parameter:
                key, value = parameter.split(separator)
                keys.append(key)
                values.append(value)
            else:
                values.append(parameter)

    return keys, values


def check_arg_with_names(arg: Tuple[List[str], List[str]]) -> bool:
    return True if len(arg[0]) == 0 and len(arg[1]) > 1 else False


def parse_args(separator: str = '=') -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.context_window_left < 0:
        parser.error('Invalid value for argument -cw_l/--context_window_left: Must be an integer >= 0.')

    if args.context_window_right < 0:
        parser.error('Invalid value for argument -cw_r/--context_window_right: Must be an integer >= 0.')

    if args.pwl_me < 0.0 or args.pwl_me > 100.0:
        parser.error('Invalid value for -pwl_me argument. It must be greater than 0.0 and less than 100.0')

    args.input = parse_arg_with_names(args.input, separator)
    if check_arg_with_names(args.input):
        parser.error(
            'Invalid format for -i/--input argment. Please specify the parameter like this '
            f'<input_name1>{separator}<file1.ark/.npz>,<input_name2>{separator}<file2.ark/.npz> or just <file.ark/.npz> in case of one input.',
        )

    args.scale_factor = parse_arg_with_names(args.scale_factor, separator)
    if check_arg_with_names(args.scale_factor):
        parser.error(
            'Invalid format for -sf/--scale_factor argment. Please specify the parameter like this '
            f'<input_name1>{separator}<sf1>,<input_name2>{separator}<sf2> or just <sf> to be applied to all inputs.',
        )

    args.output = parse_arg_with_names(args.output, separator)
    if check_arg_with_names(args.output):
        parser.error(
            'Invalid format for -o/--output argment. Please specify the parameter like this '
            f'<output_name1>{separator}<output1.ark/.npz>,<output_name2>{separator}<output2.ark/.npz> or just <output.ark/.npz> in case of one output.',
        )

    args.reference = parse_arg_with_names(args.reference, separator)
    if check_arg_with_names(args.reference):
        parser.error(
            'Invalid format for -r/--reference argment. Please specify the parameter like this '
            f'<output_name1>{separator}<reference1.ark/.npz>,<output_name2>{separator}<reference2.ark/.npz> or <reference.ark/.npz> in case of one output.',
        )

    return args
