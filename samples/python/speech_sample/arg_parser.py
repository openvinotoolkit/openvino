# -*- coding: utf-8 -*-
# Copyright (C) 2018-2022 Intel Corporation
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
    args.add_argument('-iname', '--input_layers', type=str,
                      help='Optional. Layer names for input blobs. The names are separated with ",". '
                      'Allows to change the order of input layers for -i flag. Example: Input1,Input2')
    args.add_argument('-oname', '--output_layers', type=str,
                      help='Optional. Layer names for output blobs. The names are separated with ",". '
                      'Allows to change the order of output layers for -o flag. Example: Output1:port,Output2:port.')
    args.add_argument('-cw_l', '--context_window_left', type=int, default=0,
                      help='Optional. Number of frames for left context windows (default is 0). '
                      'Works only with context window networks. '
                      'If you use the cw_l or cw_r flag, then batch size argument is ignored.')
    args.add_argument('-cw_r', '--context_window_right', type=int, default=0,
                      help='Optional. Number of frames for right context windows (default is 0). '
                      'Works only with context window networks. '
                      'If you use the cw_l or cw_r flag, then batch size argument is ignored.')
    args.add_argument('-pwl_me', type=float, default=1.0,
                      help='Optional. The maximum percent of error for PWL function. '
                      'The value must be in <0, 100> range. The default value is 1.0.')

    args = parser.parse_args()

    if args.context_window_left < 0:
        parser.error('Invalid value for argument -cw_l/--context_window_left: Must be an integer >= 0.')

    if args.context_window_right < 0:
        parser.error('Invalid value for argument -cw_r/--context_window_right: Must be an integer >= 0.')

    if args.pwl_me < 0.0 or args.pwl_me > 100.0:
        parser.error('Invalid value for -pwl_me argument. It must be greater than 0.0 and less than 100.0')

    return args
