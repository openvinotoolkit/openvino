# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
import logging
from pathlib import Path
from argparse import ArgumentParser

from openvino.tools.pot.utils.ac_imports import create_model_evaluator
from openvino.tools.pot.configs.config import Config
from openvino.tools.pot.utils.logger import init_logger, get_logger


init_logger(level=logging.DEBUG)
logger = get_logger(__name__)


def parse_args(argv):
    """Parse and process arguments for evaluation"""
    parser = ArgumentParser(description='Accuracy evaluation tool', allow_abbrev=False)
    parser.add_argument(
        '-c',
        '--config',
        help='Path to a config file with model-specific parameters',
        required=True)
    parser.add_argument(
        '-m',
        '--compressed_model',
        help='Path to a compressed model (.xml)',
        required=False)
    parser.add_argument(
        '-ss',
        '--subset_size',
        help='Size of subset to make evaluation on',
        type=int,
        required=False)
    args = parser.parse_args(args=argv)

    config = Config.read_config(args.config)
    subset = range(args.subset_size) if args.subset_size else None
    path = None
    if args.compressed_model:
        model_xml_path = Path(args.compressed_model).resolve()
        model_bin_path = model_xml_path.with_suffix('.bin')
        path = [{'model': model_xml_path, 'weights': model_bin_path}]

    logger.info('Model: %s', model_xml_path)

    return config, subset, path


def evaluate(config, subset, paths=None):
    """Evaluates performance of a compressed model using a config file"""

    # evaluate
    model_evaluator = create_model_evaluator(config.engine)
    if paths is None:
        paths = config.get_model_paths()
    model_evaluator.load_network_from_ir(paths)
    model_evaluator.process_dataset_async(check_progress=True, subset=subset)
    return model_evaluator.compute_metrics()


if __name__ == '__main__':
    _ = evaluate(*parse_args(sys.argv[1:]))
