# -*- coding: utf-8 -*-
# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import argparse


def parse_args() -> argparse.Namespace:
    """Parse and return command line arguments"""
    parser = argparse.ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    model = parser.add_mutually_exclusive_group(required=True)

    args.add_argument('-h', '--help', action='help', help='Show this help message and exit.')
    model.add_argument('-m', '--model', type=str,
                       help='Path to an .xml file with a trained model (required if -rg is missing).')
    model.add_argument('-rg', '--import_gna_model', type=str,
                       help='Read GNA model from file using path/filename provided (required if -m is missing).')
    args.add_argument('-i', '--input', required=True, type=str, help='Required. Path to an input file (.ark or .npz).')
    args.add_argument('-o', '--output', type=str,
                      help='Optional. Output file name to save inference results (.ark or .npz).')
    args.add_argument('-r', '--reference', type=str,
                      help='Optional. Read reference score file and compare scores.')
    args.add_argument('-d', '--device', default='CPU', type=str,
                      help='Optional. Specify a target device to infer on. '
                      'CPU, GPU, MYRIAD, GNA_AUTO, GNA_HW, GNA_SW_FP32, GNA_SW_EXACT and HETERO with combination of GNA'
                      ' as the primary device and CPU as a secondary (e.g. HETERO:GNA,CPU) are supported. '
                      'The sample will look for a suitable plugin for device specified. Default value is CPU.')
    args.add_argument('-bs', '--batch_size', default=1, type=int, choices=range(1, 9), metavar='[1-8]',
                      help='Optional. Batch size 1-8 (default 1).')
    args.add_argument('-qb', '--quantization_bits', default=16, type=int, choices=(8, 16), metavar='[8, 16]',
                      help='Optional. Weight bits for quantization: 8 or 16 (default 16).')
    args.add_argument('-sf', '--scale_factor', type=str,
                      help='Optional. The user-specified input scale factor for quantization. '
                      'If the network contains multiple inputs, provide scale factors by separating them with commas.')
    args.add_argument('-wg', '--export_gna_model', type=str,
                      help='Optional. Write GNA model to file using path/filename provided.')
    args.add_argument('-we', '--export_embedded_gna_model', type=str, help=argparse.SUPPRESS)
    args.add_argument('-we_gen', '--embedded_gna_configuration', default='GNA1', type=str, help=argparse.SUPPRESS)
    args.add_argument('-pc', '--performance_counter', action='store_true',
                      help='Optional. Enables performance report (specify -a to ensure arch accurate results).')
    args.add_argument('-a', '--arch', default='CORE', type=str.upper, choices=('CORE', 'ATOM'), metavar='[CORE, ATOM]',
                      help='Optional. Specify architecture. CORE, ATOM with the combination of -pc.')
    args.add_argument('-iname', '--input_layers', type=str,
                      help='Optional. Layer names for input blobs. The names are separated with ",". '
                      'Allows to change the order of input layers for -i flag. Example: Input1,Input2')
    args.add_argument('-oname', '--output_layers', type=str,
                      help='Optional. Layer names for output blobs. The names are separated with ",". '
                      'Allows to change the order of output layers for -o flag. Example: Output1:port,Output2:port.')
    args.add_argument('-cw_l', '--context_window_left', type=IntRange(0), default=0,
                      help='Optional. Number of frames for left context windows (default is 0). '
                      'Works only with context window networks. '
                      'If you use the cw_l or cw_r flag, then batch size argument is ignored.')
    args.add_argument('-cw_r', '--context_window_right', type=IntRange(0), default=0,
                      help='Optional. Number of frames for right context windows (default is 0). '
                      'Works only with context window networks. '
                      'If you use the cw_l or cw_r flag, then batch size argument is ignored.')

    return parser.parse_args()


class IntRange:
    """Custom argparse type representing a bounded int."""

    def __init__(self, _min=None, _max=None):
        self._min = _min
        self._max = _max

    def __call__(self, arg):
        try:
            value = int(arg)
        except ValueError:
            raise argparse.ArgumentTypeError('Must be an integer.')

        if (self._min is not None and value < self._min) or (self._max is not None and value > self._max):
            if self._min is not None and self._max is not None:
                raise argparse.ArgumentTypeError(f'Must be an integer in the range [{self._min}, {self._max}].')
            elif self._min is not None:
                raise argparse.ArgumentTypeError(f'Must be an integer >= {self._min}.')
            elif self._max is not None:
                raise argparse.ArgumentTypeError(f'Must be an integer <= {self._max}.')

        return value
