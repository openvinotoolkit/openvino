"""
Copyright (c) 2019 Intel Corporation

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

from pathlib import Path
from argparse import ArgumentParser
from functools import partial

from .config import ConfigReader
from .logging import print_info, add_file_handler
from .evaluators import ModelEvaluator, PipeLineEvaluator, get_processing_info
from .progress_reporters import ProgressReporter
from .utils import get_path


def build_arguments_parser():
    parser = ArgumentParser(description='NN Validation on Caffe and IE', allow_abbrev=False)
    parser.add_argument(
        '-d', '--definitions',
        help='path to the yml file with definitions',
        type=get_path,
        required=False
    )
    parser.add_argument(
        '-c', '--config',
        help='path to the yml file with local configuration',
        type=get_path,
        required=True
    )
    parser.add_argument(
        '-m', '--models',
        help='prefix path to the models and weights',
        type=partial(get_path, is_directory=True),
        default=Path.cwd(),
        required=False
    )
    parser.add_argument(
        '-s', '--source',
        help='prefix path to the data source',
        type=partial(get_path, is_directory=True),
        default=Path.cwd(),
        required=False
    )
    parser.add_argument(
        '-a', '--annotations',
        help='prefix path to the converted annotations and datasets meta data',
        type=partial(get_path, is_directory=True),
        default=Path.cwd(),
        required=False
    )
    parser.add_argument(
        '-e', '--extensions',
        help='prefix path to extensions folder',
        type=partial(get_path, is_directory=True),
        default=Path.cwd(),
        required=False
    )
    parser.add_argument(
        '--cpu_extensions_mode',
        help='specified preferable set of processor instruction for automatic searching cpu extension lib',
        required=False,
        choices=['avx512', 'avx2', 'sse4']
    )
    parser.add_argument(
        '-b', '--bitstreams',
        help='prefix path to bitstreams folder',
        type=partial(get_path, is_directory=True),
        default=Path.cwd(),
        required=False
    )
    parser.add_argument(
        '--stored_predictions',
        help='path to file with saved predictions. Used for development',
        # since at the first time file does not exist and then created we can not always check existence
        required=False
    )
    parser.add_argument(
        '-C', '--converted_models',
        help='directory to store Model Optimizer converted models. Used for DLSDK launcher only',
        type=partial(get_path, is_directory=True),
        default=Path.cwd(),
        required=False
    )
    parser.add_argument(
        '-M', '--model_optimizer',
        help='path to model optimizer directory',
        type=partial(get_path, is_directory=True),
        # there is no default value because if user did not specify it we use specific locations
        # defined in model_conversion.py
        required=False
    )
    parser.add_argument(
        '--tf_custom_op_config_dir',
        help='path to directory with tensorflow custom operation configuration files for model optimizer',
        type=partial(get_path, is_directory=True),
        # there is no default value because if user did not specify it we use specific location
        # defined in model_conversion.py
        required=False
    )
    parser.add_argument(
        '--tf_obj_detection_api_pipeline_config_path',
        help='path to directory with tensorflow object detection api pipeline configuration files for model optimizer',
        type=partial(get_path, is_directory=True),
        # there is no default value because if user did not specify it we use specific location
        # defined in model_conversion.py
        required=False
    )
    parser.add_argument(
        '--progress',
        help='progress reporter',
        required=False,
        default='bar'
    )
    parser.add_argument(
        '-tf', '--target_framework',
        help='framework for infer',
        required=False
    )
    parser.add_argument(
        '-td', '--target_devices',
        help='Space separated list of devices for infer',
        required=False,
        nargs='+'
    )

    parser.add_argument(
        '-tt', '--target_tags',
        help='Space separated list of launcher tags for infer',
        required=False,
        nargs='+'
    )

    parser.add_argument(
        '-l', '--log_file',
        help='file for additional logging results',
        required=False
    )

    parser.add_argument(
        '--ignore_result_formatting',
        help='allow to get raw metrics results without data formatting',
        required=False,
        default=False
    )

    parser.add_argument(
        '-am', '--affinity_map',
        help='prefix path to the affinity maps',
        type=partial(get_path, is_directory=True),
        default=Path.cwd(),
        required=False
    )

    parser.add_argument(
        '--aocl',
        help='path to aocl executable for FPGA bitstream programming',
        type=get_path,
        required=False
    )
    parser.add_argument(
        '--vpu_log_level',
        help='log level for VPU devices',
        required=False,
        choices=['LOG_NONE', 'LOG_WARNING', 'LOG_INFO', 'LOG_DEBUG'],
        default='LOG_WARNING'
    )

    return parser


def main():
    args = build_arguments_parser().parse_args()
    progress_reporter = ProgressReporter.provide((
        args.progress if ':' not in args.progress
        else args.progress.split(':')[0]
    ))
    if args.log_file:
        add_file_handler(args.log_file)

    config, mode = ConfigReader.merge(args)
    if mode == 'models':
        model_evaluation_mode(config, progress_reporter, args)
    else:
        pipeline_evaluation_mode(config, progress_reporter, args)


def model_evaluation_mode(config, progress_reporter, args):
    for model in config['models']:
        for launcher_config in model['launchers']:
            for dataset_config in model['datasets']:
                print_processing_info(
                    model['name'],
                    launcher_config['framework'],
                    launcher_config['device'],
                    launcher_config.get('tags'),
                    dataset_config['name']
                )
                model_evaluator = ModelEvaluator.from_configs(launcher_config, dataset_config)
                progress_reporter.reset(model_evaluator.dataset.size)
                model_evaluator.process_dataset(args.stored_predictions, progress_reporter=progress_reporter)
                model_evaluator.compute_metrics(ignore_results_formatting=args.ignore_result_formatting)

                model_evaluator.release()


def pipeline_evaluation_mode(config, progress_reporter, args):
    for pipeline_config in config['pipelines']:
        print_processing_info(*get_processing_info(pipeline_config))
        evaluator = PipeLineEvaluator.from_configs(pipeline_config['stages'])
        evaluator.process_dataset(args.stored_predictions, progress_reporter=progress_reporter)
        evaluator.compute_metrics(ignore_results_formatting=args.ignore_result_formatting)

        evaluator.release()


def print_processing_info(model, launcher, device, tags, dataset):
    print_info('Processing info:')
    print_info('model: {}'.format(model))
    print_info('launcher: {}'.format(launcher))
    if tags:
        print_info('launcher tags: {}'.format(' '.join(tags)))
    print_info('device: {}'.format(device))
    print_info('dataset: {}'.format(dataset))


if __name__ == '__main__':
    main()
