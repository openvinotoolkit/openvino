# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import os

from openvino.tools.pot import load_model, save_model, create_pipeline
from openvino.tools.pot.utils.logger import init_logger
from openvino.tools.pot.api.samples.utils.argument_parser import get_common_argparser
from openvino.tools.pot.api.samples.speech.data_loader import ArkDataLoader
from openvino.tools.pot.engines.simplified_engine import SimplifiedEngine
from openvino.tools.pot.graph.model_utils import compress_model_weights


def parse_args():
    parser = get_common_argparser()
    parser.add_argument(
        '-i',
        '--input_names',
        help='List of input names of network',
        required=True
    )
    parser.add_argument(
        '-f',
        '--files_for_input',
        help='List of filenames mapped to input names (without .ark extension)',
        required=True
    )
    parser.add_argument(
        '-is',
        '--input_shapes',
        help='List of input shapes mapped to input names of network. Input data will be reshaped with provided shape. '
             'Use this argument only for 3D shape inputs. '
             'List example: [[1,2,3],[4,5,6]]',
        type=json.loads,
        default=[]
    )
    parser.add_argument(
        '-p',
        '--preset',
        help='Preset for quantization. '
             '-performance for INT8 weights and INT16 inputs; '
             '-accuracy for INT16 weights and inputs',
        default='accuracy',
        choices=['performance', 'accuracy'])
    parser.add_argument(
        '-o',
        '--output',
        help='Path to save the quantized model',
        default='./model/optimized')
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'],
        help='Log level to print')
    parser.add_argument(
        '-s',
        '--subset_size',
        help='Subset size for calibration',
        default=2000,
        type=int)
    parser.add_argument(
        '-q',
        '--write_quantized_weights_to_bin_file',
        help='Write the quantized weights as INT8 to the output IR bin file',
        action="store_true")
    return parser.parse_args()


def get_configs(args):
    if not args.weights:
        args.weights = '{}.bin'.format(os.path.splitext(args.model)[0])

    model_config = {
        'model_name': 'gna_model',
        'model': os.path.expanduser(args.model),
        'weights': os.path.expanduser(args.weights),
        'exec_log_dir': args.output
    }
    engine_config = {
        'device': 'CPU',
        'stat_requests_number': 1,
        'eval_requests_number': 1
    }
    dataset_config = {
        'data_source': os.path.expanduser(args.dataset),
        'type': 'simplified',
        'input_names': args.input_names.split(','),
        'input_shapes': args.input_shapes
    }

    if args.files_for_input is not None:
        dataset_config['input_files'] = args.files_for_input.split(',')

    algorithms = [
        {
            'name': 'DefaultQuantization',
            'params': {
                'target_device': 'GNA',
                'preset': args.preset,
                # The custom configuration is for speech recognition models
                'stat_subset_size': args.subset_size,
                'activations': {
                    'range_estimator': {
                        'max': {
                            'type': 'abs_max',
                            'aggregator': 'max'
                        }
                    }
                }
            }
        }
    ]

    return model_config, engine_config, dataset_config, algorithms


def optimize_model(args):
    model_config, engine_config, dataset_config, algorithms = get_configs(args)
    data_loader = ArkDataLoader(dataset_config)
    engine = SimplifiedEngine(config=engine_config, data_loader=data_loader)
    pipeline = create_pipeline(algorithms, engine)

    model = load_model(model_config, target_device='GNA')
    return pipeline.run(model)


def main():
    args = parse_args()
    out_dir = os.path.expanduser(args.output)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    init_logger(level=args.log_level, file_name=os.path.join(out_dir, 'log.txt'))
    compressed_model = optimize_model(args)

    if args.write_quantized_weights_to_bin_file:
        compress_model_weights(model=compressed_model)

    save_model(compressed_model, out_dir)


if __name__ == '__main__':
    main()
