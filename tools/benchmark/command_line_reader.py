"""
Copyright (C) 2018-2019 Intel Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import collections
import errno
import pathlib
from functools import partial
from argparse import ArgumentParser
from typing import Union

from ..accuracy_checker.accuracy_checker.config import ConfigReader
from ..accuracy_checker.accuracy_checker.utils import get_path
from ..network import Network

from .configuration import Configuration
from .logging import info


class CommandLineReader:
    """
    Class for parsing input config
    """
    @staticmethod
    def read():
        args, unknown_args = CommandLineReader.__build_arguments_parser().parse_known_args()
        if unknown_args:
            info("unknown command line arguments: {0}".format(unknown_args))

        args.target_framework = "dlsdk"
        args.aocl = None

        merged_config = ConfigReader.merge(args)
        launcher = merged_config['models'][0]['launchers'][0]

        batch_size = args.batch_size if args.batch_size else (launcher['batch'] if 'batch' in launcher else None)
        if not batch_size:
            with Network(str(launcher['model']), str(launcher['weights'])) as network:
                batch_size = network.ie_network.batch_size

        return Configuration(
            config = merged_config,
            model = str(launcher['model']),
            weights = str(launcher['weights']),
            cpu_extension = (str(launcher['cpu_extensions']) if 'cpu_extensions' in launcher else None),
            gpu_extension = (str(launcher['gpu_extensions']) if 'gpu_extensions' in launcher else None),
            device = launcher['device'],
            benchmark_iterations_count = args.benchmark_iterations_count)

    @staticmethod
    def __build_arguments_parser():
        parser = ArgumentParser(description='openvino.tools.benchmark')

        parser.add_argument(
            '-d', '--definitions',
            help='Optional. Path to the YML file with definitions',
            type=str,
            required=False)

        parser.add_argument(
            '-c',
            '--config',
            help='Required. Path to the YML file with local configuration',
            type=get_path,
            required=True)

        parser.add_argument(
            '-m', '--models',
            help='Optional. Prefix path to the models and weights',
            type=partial(get_path, is_directory=True),
            default=pathlib.Path.cwd(),
            required=False)

        parser.add_argument(
            '-s', '--source',
            help='Optional. prefix path to the data source',
            type=partial(get_path, is_directory=True),
            default=pathlib.Path.cwd(),
            required=False)

        parser.add_argument(
            '-a', '--annotations',
            help='Optional. prefix path to the converted annotations and datasets meta data',
            type=partial(get_path, is_directory=True),
            default=pathlib.Path.cwd(),
            required=False)

        parser.add_argument(
            '-e', '--extensions',
            help='Optional. Prefix path to extensions folder',
            type=partial(get_path, is_directory=True),
            default=pathlib.Path.cwd(),
            required=False)

        parser.add_argument(
            '--cpu_extensions_mode',
            help='Optional. specified preferable set of processor instruction for automatic searching cpu extension lib',
            required=False,
            choices=['avx2', 'sse4'])

        parser.add_argument(
            '-b', '--bitstreams',
            help='Optional. prefix path to bitstreams folder',
            type=partial(get_path, is_directory=True),
            default=pathlib.Path.cwd(),
            required=False)

        parser.add_argument(
            '-C', '--converted_models', '--converted-models',
            help='Optional. directory to store Model Optimizer converted models. Used for DLSDK launcher only',
            type=partial(get_path, is_directory=True),
            default=pathlib.Path.cwd(),
            required=False)

        parser.add_argument(
            '-td', '--target_devices', '--target-devices',
            help='Optional. Space-separated list of devices for infer',
            required=False,
            nargs='+',
            default=["CPU"])

        parser.add_argument(
            '-tt', '--target_tags', '--target-tags',
            help='Optional. Space-separated list of launcher tags for infer',
            required=False,
            nargs='+')

        parser.add_argument(
            '--batch-size',
            help='Optional. Batch size value. If not specified, the batch size value is determined from IR',
            type=int,
            required=False)

        parser.add_argument(
            '-ic',
            '--benchmark_iterations_count',
            help='Optional. Benchmark itertations count. (1000 is default)',
            type=float,
            required=False,
            default=1000)

        return parser