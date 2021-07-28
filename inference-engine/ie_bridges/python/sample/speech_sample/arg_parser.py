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
    args.add_argument('-bs', '--batch_size', default=1, type=int, help='Optional. Batch size 1-8 (default 1).')
    args.add_argument('-qb', '--quantization_bits', default=16, type=int,
                      help='Optional. Weight bits for quantization: 8 or 16 (default 16).')
    args.add_argument('-wg', '--export_gna_model', type=str,
                      help='Optional. Write GNA model to file using path/filename provided.')
    args.add_argument('-we', '--export_embedded_gna_model', type=str, help=argparse.SUPPRESS)
    args.add_argument('-we_gen', '--embedded_gna_configuration', default='GNA1', type=str, help=argparse.SUPPRESS)
    args.add_argument('-iname', '--input_layers', type=str,
                      help='Optional. Layer names for input blobs. The names are separated with ",". '
                      'Allows to change the order of input layers for -i flag. Example: Input1,Input2')
    args.add_argument('-oname', '--output_layers', type=str,
                      help='Optional. Layer names for output blobs. The names are separated with ",". '
                      'Allows to change the order of output layers for -o flag. Example: Output1:port,Output2:port.')

    return parser.parse_args()
