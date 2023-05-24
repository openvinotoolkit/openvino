# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
from datetime import datetime

import os
from pathlib import Path

from openvino.tools.pot.app.argparser import get_common_argument_parser, check_dependencies
from openvino.tools.pot.configs.config import Config
from openvino.tools.pot.data_loaders.creator import create_data_loader
from openvino.tools.pot.engines.creator import create_engine
from openvino.tools.pot.graph import load_model, save_model
from openvino.tools.pot.graph.model_utils import compress_model_weights
from openvino.tools.pot.pipeline.initializer import create_pipeline
from openvino.tools.pot.utils.logger import init_logger, get_logger
from openvino.tools.pot.utils.telemetry import start_session_telemetry, end_session_telemetry

logger = get_logger(__name__)

_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


def main():
    app(sys.argv[1:])


def app(argv):
    logger.warning('Post-training Optimization Tool is deprecated and will be removed in the future.'
                   ' Please use Neural Network Compression Framework'
                   ' instead: https://github.com/openvinotoolkit/nncf')
    telemetry = start_session_telemetry()
    parser = get_common_argument_parser()
    args = parser.parse_args(args=argv)
    check_dependencies(args)
    if not args.config:
        _update_config_path(args)

    config = Config.read_config(args.config)

    if args.engine:
        config.engine['type'] = args.engine if args.engine else 'accuracy_checker'
    if 'data_source' not in config.engine:
        config.engine['data_source'] = args.data_source

    config.configure_params(args.ac_config)
    config.update_from_args(args)

    if config.engine.type != 'accuracy_checker' and args.evaluate:
        raise Exception('Can not make evaluation in simplified mode')

    log_dir = _create_log_path(config)
    init_logger(level=args.log_level,
                file_name=os.path.join(log_dir, 'log.txt'),
                progress_bar=args.pbar)
    logger.info('Output log dir: {}'.format(log_dir))

    metrics = optimize(config)
    if metrics and logger.progress_bar_disabled:
        for name, value in metrics.items():
            logger.info('{: <27s}: {}'.format(name, value))
    end_session_telemetry(telemetry)


def _create_log_path(config):
    if config.model.direct_dump:
        model_log_dir = config.model.output_dir
        exec_log_dir = config.model.output_dir
    else:
        model_log_dir = os.path.join(config.model.output_dir, config.model.log_algo_name)
        exec_log_dir = os.path.join(model_log_dir, _timestamp)
    config.add_log_dir(model_log_dir, exec_log_dir)

    if not os.path.isdir(exec_log_dir):
        os.makedirs(exec_log_dir)

    return exec_log_dir


def _update_config_path(args):
    config_template_folder = os.path.join(Path(__file__).parents[1], 'configs', 'templates')

    if args.quantize is not None:
        if args.quantize == 'default':
            args.config = os.path.join(config_template_folder, 'default_quantization_template.json')
        elif args.quantize == 'accuracy_aware':
            args.config = os.path.join(config_template_folder, 'accuracy_aware_quantization_template.json')


def print_algo_configs(config):
    # log algorithms settings
    configs_string = 'Creating pipeline:'
    for algo in config:
        configs_string += '\n Algorithm: {}'.format(algo.name)
        configs_string += '\n Parameters:'
        for name, value in algo.params.items():
            configs_string += '\n\t{: <27s}: {}'.format(name, value)
    configs_string += '\n {}'.format('=' * 75)
    logger.info(configs_string)


def optimize(config):
    """Creates pipeline of compression algorithms and optimize its parameters"""

    if logger.progress_bar_disabled:
        print_algo_configs(config.compression.algorithms)

    # load custom model
    model = load_model(config.model, target_device=config.compression.target_device)

    data_loader = None
    # create custom data loader in case of custom Engine
    if config.engine.type != 'accuracy_checker':
        data_loader = create_data_loader(config.engine, model)

    engine = create_engine(config.engine, data_loader=data_loader, metric=None)

    pipeline = create_pipeline(
        config.compression.algorithms, engine, 'CLI')

    compressed_model = pipeline.run(model)

    if not config.model.keep_uncompressed_weights:
        compress_model_weights(compressed_model)

    save_model(compressed_model,
               os.path.join(config.model.exec_log_dir, 'optimized'),
               model_name=config.model.model_name)

    # evaluating compressed model if need
    if config.engine.evaluate:
        return pipeline.evaluate(compressed_model)

    return None
