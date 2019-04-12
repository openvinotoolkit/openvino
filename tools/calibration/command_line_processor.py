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

import tempfile

from ..accuracy_checker.accuracy_checker.config import ConfigReader
from ..accuracy_checker.accuracy_checker.launcher.dlsdk_launcher import DLSDKLauncher

from ..network import Network
from ..utils.path import Path
from ..utils.configuration_filter import ConfigurationFilter
from .calibration_configuration import CalibrationConfiguration
from .logging import info, default_logger
from .command_line_reader import CommandLineReader


class CommandLineProcessor:
    """
    Class for parsing user input config
    """
    @staticmethod
    def process() -> CalibrationConfiguration:
        args, unknown_args = CommandLineReader.parser().parse_known_args()
        if unknown_args:
            info("unknown command line arguments: {0}".format(unknown_args))

        args.target_framework = "dlsdk"
        args.aocl = None

        merged_config = ConfigReader.merge(args)
        updated_config = ConfigurationFilter.filter(merged_config, args.metric_name, args.metric_type, default_logger)

        if len(updated_config['models']) > 1:
            raise ValueError("too much models")

        if len(updated_config['models'][0]['launchers']) > 1:
            raise ValueError("too much launchers")

        launcher = updated_config['models'][0]['launchers'][0]
        if 'caffe_model' in launcher or 'tf_model' in launcher or 'mxnet_weights' in launcher:
            if args.converted_models:
                tmp_directory = None
            else:
                tmp_directory = tempfile.mkdtemp(".converted_models")
                launcher['mo_params']['output_dir'] = tmp_directory

            if 'caffe_model' in launcher:
                framework = 'caffe'
                output_model = Path.get_model(
                    str(launcher['caffe_model']),
                    "_i8",
                    str(args.output_dir) if args.output_dir else None)
                output_weights = Path.get_weights(
                    str(launcher['caffe_weights']),
                    "_i8",
                    str(args.output_dir) if args.output_dir else None)
            elif 'tf_model' in launcher:
                framework = 'tf'
                output_model = Path.get_model(
                    str(launcher['tf_model']),
                    "_i8",
                    str(args.output_dir) if args.output_dir else None)
                output_weights = Path.get_weights(
                    str(launcher['tf_model']),
                    "_i8",
                    str(args.output_dir) if args.output_dir else None)
            elif 'mxnet_weights' in launcher:
                framework = 'mxnet'
                output_model = Path.get_model(
                    str(launcher['mxnet_weights']),
                    "_i8",
                    str(args.output_dir) if args.output_dir else None)
                output_weights = Path.get_weights(
                    str(launcher['mxnet_weights']),
                    "_i8",
                    str(args.output_dir) if args.output_dir else None)
            else:
                raise ValueError("unknown model framework")

            model, weights = DLSDKLauncher.convert_model(launcher, framework)
            launcher['model'] = model
            launcher['weights'] = weights

            launcher.pop('caffe_model', None)
            launcher.pop('caffe_weights', None)
            launcher.pop('tf_model', None)
            launcher.pop('mxnet_weights', None)
        else:
            model = launcher['model']
            output_model = Path.get_model(str(model), "_i8", str(args.output_dir) if args.output_dir else None)
            weights = launcher['weights']
            output_weights = Path.get_weights(str(weights), "_i8", str(args.output_dir) if args.output_dir else None)
            tmp_directory = None

        batch_size = args.batch_size if args.batch_size else (launcher['batch'] if 'batch' in launcher else None)
        if not batch_size:
            with Network(str(launcher['model']), str(launcher['weights'])) as network:
                batch_size = network.ie_network.batch_size

        if 'cpu_extensions' in launcher:
            cpu_extension = DLSDKLauncher.get_cpu_extension(launcher['cpu_extensions'], args.cpu_extensions_mode)
            launcher['cpu_extensions'] = cpu_extension
        else:
            cpu_extension = None

        if not args.calibrate_fully_connected:
            if args.ignore_layer_types is None:
                args.ignore_layer_types = []
            args.ignore_layer_types.append("FullyConnected")

        return CalibrationConfiguration(
            config=updated_config,
            precision=args.precision,
            model=str(model),
            weights=str(weights),
            tmp_directory=tmp_directory,
            output_model=output_model,
            output_weights=output_weights,
            cpu_extension=str(cpu_extension) if cpu_extension else None,
            gpu_extension=str(launcher['gpu_extensions']) if 'gpu_extensions' in launcher else None,
            device=launcher['device'],
            batch_size=batch_size,
            threshold=args.threshold,
            ignore_layer_types=args.ignore_layer_types,
            ignore_layer_types_path=args.ignore_layer_types_path,
            ignore_layer_names=args.ignore_layer_names,
            ignore_layer_names_path=args.ignore_layer_names_path,
            benchmark_iterations_count=args.benchmark_iterations_count,
            progress=(None if args.progress == 'None' else args.progress))