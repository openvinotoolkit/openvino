# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from argparse import ArgumentParser


def get_common_argument_parser():
    """Defines command-line arguments, and parses them.

    """
    parser = ArgumentParser(description='Post-training Compression Toolkit', allow_abbrev=False)

    parser.add_argument(
        '-c',
        '--config',
        help='Path to a config file with optimization parameters. '
             'Overrides "-q | -m | -w | --ac-config | --engine" options')

    parser.add_argument(
        '-q',
        '--quantize',
        type=str,
        choices=['default', 'accuracy_aware'],
        help='Quantize model to 8 bits with specified quantization method: default | accuracy_aware')

    parser.add_argument(
        '--preset',
        type=str,
        choices=['performance', 'mixed'],
        help='Use "performance" for fully symmetric quantization or "mixed" preset for symmetric quantization of '
             'weight and asymmetric quantization of activations. Applicable only when -q option is used.')

    parser.add_argument(
        '-m',
        '--model',
        type=str,
        help='Path to the model file (.xml). Applicable only when -q option is used.')

    parser.add_argument(
        '-w',
        '--weights',
        type=str,
        help='Path to the weights file (.bin). Applicable only when -q option is used.')

    parser.add_argument(
        '--name',
        '-n',
        type=str,
        help='Model name. Applicable only when -q option is used.')

    parser.add_argument(
        '--engine',
        choices=['accuracy_checker', 'data_free', 'simplified'],
        type=str,
        help='Engine type. Default: `accuracy_checker`')

    parser.add_argument(
        '--ac-config',
        type=str,
        help='Path to the Accuracy Checker configuration file. Applicable only when -q option is used.')

    parser.add_argument(
        '--max-drop',
        type=float,
        help='Maximum accuracy drop. Valid only for accuracy-aware quantization. '
             'Applicable only when -q option is used.')

    parser.add_argument(
        '-e',
        '--evaluate',
        action='store_true',
        help='Whether to evaluate model on whole dataset')

    # Storage settings
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results',
        help='The directory where models are saved. Default: ./results')

    parser.add_argument(
        '-d',
        '--direct-dump',
        action='store_true',
        help='Flag to save files without sub folders with algo names')

    log_levels = ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG']
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=log_levels,
        help='Log level to print')

    parser.add_argument(
        '--progress-bar',
        dest='pbar',
        action='store_true',
        default=False,
        help='Disable CL logging and enable progress bar')

    parser.add_argument(
        '--stream-output',
        dest='stream_output',
        action='store_true',
        default=False,
        help='Switch progress display to a multiline mode')

    parser.add_argument(
        '--keep-uncompressed-weights',
        action='store_true',
        default=False,
        help='Keep Convolution, Deconvolution and FullyConnected weights uncompressed')

    parser.add_argument(
        '--data-source',
        help='Valid for DataFree and Simplified modes. For Simplified mode path to dataset dir is required. '
             'For DataFree mode specify path to directory '
             'where syntetic dataset is located or will be generated and saved. '
             'For DataFree mode default: `./pot_dataset`')

    data_free_opt = parser.add_argument_group('DataFree mode options')

    data_free_opt.add_argument(
        '--shape',
        type=str,
        help='Required for models with dynamic shapes. '
             'Input shape that should be fed to an input node of the model. '
             'Shape is defined as a comma-separated list of integer numbers enclosed in '
             'parentheses or square brackets, for example [1,3,227,227] or (1,227,227,3), where '
             'the order of dimensions depends on the framework input layout of the model.')

    data_free_opt.add_argument(
        '--data-type',
        type=str,
        default='image',
        choices=['image'],
        help='Type of data for generation. Dafault: `image`')

    data_free_opt.add_argument(
        '--generate-data',
        action='store_true',
        default=False,
        help='If specified, generate synthetic data and store to `data-source`. '
             'Otherwise, the dataset from `--data-source` will be used')

    return parser


def check_dependencies(args):
    if args.quantize is None and args.config is None:
        raise ValueError(
            'Either --config or --quantize option should be specified')
    if args.quantize is not None and args.config is not None:
        raise ValueError('Either --config or --quantize option should be specified')
    if args.quantize is not None and (args.model is None or args.weights is None):
        raise ValueError(
            '--quantize option requires model and weights to be specified.')
    if (args.quantize is not None and args.ac_config is None and
            (args.engine == 'accuracy_checker' or args.engine is None)):
        raise ValueError(
            '--quantize option requires AC config to be specified '
            'or --engine should be `data_free` or `simplified`.')
    if args.quantize == 'accuracy_aware' and args.max_drop is None:
        raise ValueError('For AccuracyAwareQuantization --max-drop should be specified')
    if args.config is None and args.engine == 'simplified' and args.data_source is None:
        raise ValueError('For Simplified mode `--data-source` option should be specified')
    check_extra_arguments(args, 'model')
    check_extra_arguments(args, 'weights')
    check_extra_arguments(args, 'preset')
    check_extra_arguments(args, 'ac_config')


def check_extra_arguments(args, parameter):
    if args.config is not None and vars(args)[parameter] is not None:
        raise ValueError('Either --config or --{} option should be specified'.format(parameter))
