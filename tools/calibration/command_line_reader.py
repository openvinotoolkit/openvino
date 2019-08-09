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

import pathlib
from functools import partial
from argparse import ArgumentParser

from ..accuracy_checker.accuracy_checker.utils import get_path
from ..utils.path import Path

class CommandLineReader:
    @staticmethod
    def parser():
        parser = ArgumentParser(description='openvino.tools.calibration')

        parser.add_argument(
            '-d', '--definitions',
            help='Optional. Path to the YML file with definitions',
            type=str,
            required=False)

        parser.add_argument(
            '-c', '--config',
            help='Optional. Path to the YML file with local configuration',
            type=get_path,
            required=False)

        parser.add_argument(
            '-m', '--models',
            help='Optional. Prefix path to the models and weights. In the simplified mode, it is the path to IR .xml file',
            type=Path.validate_path,
            default=pathlib.Path.cwd(),
            required=False)

        parser.add_argument(
            '-s', '--source',
            help='Optional. Prefix path to the data source. In the simplified mode, it is the path to a folder with images',
            type=partial(get_path, is_directory=True),
            default=pathlib.Path.cwd(),
            required=False)

        parser.add_argument(
            '-a', '--annotations',
            help='Optional. Prefix path to the converted annotations and datasets meta data',
            type=partial(get_path, is_directory=True),
            default=pathlib.Path.cwd(),
            required=False)

        parser.add_argument(
            '-e', '--extensions',
            help='Optional. Prefix path to extensions folder. In simplified mode is a path to extensions library',
            type=Path.validate_path,
            default=pathlib.Path.cwd(),
            required=False)

        parser.add_argument(
            '--cpu_extensions_mode', '--cpu-extensions-mode',
            help='Optional. specified preferable set of processor instruction for automatic searching cpu extension lib',
            required=False,
            choices=['avx2', 'sse4'])

        parser.add_argument(
            '-C', '--converted_models', '--converted-models',
            help='Optional. Directory to store Model Optimizer converted models. Used for DLSDK launcher only',
            type=partial(get_path, is_directory=True),
            required=False
        )

        parser.add_argument(
            '-M', '--model_optimizer', '--model-optimizer',
            help='Optional. Path to model optimizer caffe directory',
            type=partial(get_path, is_directory=True),
            # there is no default value because if user did not specify it we use specific locations
            # defined in model_conversion.py
            required=False
        )

        parser.add_argument(
            '--tf_custom_op_config_dir', '--tf-custom-op-config-dir',
            help='Optional. Path to directory with tensorflow custom operation configuration files for model optimizer',
            type=partial(get_path, is_directory=True),
            # there is no default value because if user did not specify it we use specific location
            # defined in model_conversion.py
            required=False
        )

        parser.add_argument(
            '--tf_obj_detection_api_pipeline_config_path', '--tf-obj-detection-api-pipeline-config-path',
            help='Optional. Path to directory with tensorflow object detection api pipeline configuration files for model optimizer',
            type=partial(get_path, is_directory=True),
            # there is no default value because if user did not specify it we use specific location
            # defined in model_conversion.py
            required=False
        )

        parser.add_argument(
            '--progress',
            help='Optional. Progress reporter',
            required=False,
            default='bar')

        parser.add_argument(
            '-td', '--target_devices', '--target-devices',
            help='Optional. Space-separated list of devices for infer',
            required=False,
            nargs='+',
            default=["CPU"]
        )

        parser.add_argument(
            '-tt', '--target_tags', '--target-tags',
            help='Optional. Space-separated list of launcher tags for infer',
            required=False,
            nargs='+')

        parser.add_argument(
            '-p',
            '--precision',
            help='Optional. Precision to calibrate. Default value is INT8. '
                 'In simplified mode determines output IR precision',
            type=str,
            required=False,
            default='INT8')

        parser.add_argument(
            '--ignore_layer_types', '--ignore-layer-types',
            help='Optional. Layer types list which will be skipped during quantization',
            type=str,
            required=False,
            nargs='+')

        parser.add_argument(
            '--ignore_layer_types_path', '--ignore-layer-types-path',
            help='Optional. Ignore layer types file path',
            type=str,
            required=False,
            nargs='+')

        parser.add_argument(
            '--ignore_layer_names', '--ignore-layer-names',
            help='Optional. Layer names list which will be skipped during quantization',
            type=str,
            required=False,
            nargs='+')

        parser.add_argument(
            '--ignore_layer_names_path', '--ignore-layer-names-path',
            help='Optional. Ignore layer names file path',
            type=str,
            required=False)

        parser.add_argument(
            '--batch_size', '--batch-size',
            help='Optional. Batch size value. If not specified, the batch size value is determined from IR',
            type=int,
            required=False)

        parser.add_argument(
            '-th', '--threshold',
            help='Optional. Accuracy drop of quantized model should not exceed this threshold. '
                 'Should be pointer in percents without percent sign. (1%% is default)',
            type=float,
            required=False,
            default=1.0)

        parser.add_argument(
            '-ic', '--benchmark_iterations_count', '--benchmark-iterations-count',
            help='Optional. Benchmark itertations count. (1 is default)',
            type=int,
            required=False,
            default=1)

        parser.add_argument(
            '-mn', '--metric_name', '--metric-name',
            help='Optional. Metric name used during calibration',
            type=str,
            required=False)

        parser.add_argument(
            '-mt', '--metric_type', '--metric-type',
            help='Optional. Metric type used during calibration',
            type=str,
            required=False)

        parser.add_argument(
            '-o', '--output_dir', '--output-dir',
            help='Optional. Directory to store converted models. Original model directory is used if not defined',
            type=partial(get_path, is_directory=True),
            required=False)

        parser.add_argument(
            '-cfc', '--calibrate_fully_connected', '--calibrate-fully-connected',
            help='Optional. FullyConnected INT8 convertion support (False is default)',
            action="store_true",
            required=False)

        parser.add_argument(
            '-thstep', '--threshold_step', '--threshold-step',
            help='Optional. Activation statistics threshold step',
            type=float,
            required=False,
            default=0.5
        )

        parser.add_argument(
            '-thboundary', '--threshold_boundary', '--threshold-boundary',
            help='Optional. Activation statistics lower boundary',
            type=float,
            required=False,
            default=95.0
        )

        parser.add_argument(
            '-sm', '--simplified_mode', '--simplified-mode',
            help='Optional. If specified, calibration tool will just collect statistics without searching optimal data thresholds.',
            action="store_true",
            required=False
        )

        parser.add_argument(
            '-ss', '--subset',
            help='Optional. This option is used just with --simplified_mode. '
                 'Specifies number of images from folder set via -s option.',
            type=int,
            required=False,
            default=0
        )

        return parser
