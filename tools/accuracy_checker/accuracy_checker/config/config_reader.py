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

import copy
from pathlib import Path

import warnings

from ..utils import read_yaml, to_lower_register, contains_any
from .config_validator import ConfigError


class ConfigReader:
    """
    Class for parsing input config.
    """

    @staticmethod
    def merge(arguments):
        """
        Args:
            arguments: command-line arguments.
        Returns:
            dictionary containing configuration.
        """

        global_config, local_config = ConfigReader._read_configs(arguments)
        if not local_config:
            raise ConfigError('Missing local config')

        mode = ConfigReader._check_local_config(local_config)
        ConfigReader._prepare_global_configs(global_config)

        config = ConfigReader._merge_configs(global_config, local_config, mode)

        ConfigReader._provide_cmd_arguments(arguments, config, mode)
        ConfigReader._merge_paths_with_prefixes(arguments, config, mode)
        ConfigReader._filter_launchers(config, arguments, mode)

        return config, mode

    @staticmethod
    def _read_configs(arguments):
        global_config = read_yaml(arguments.definitions) if arguments.definitions else None
        local_config = read_yaml(arguments.config)

        return global_config, local_config

    @staticmethod
    def _check_local_config(config):
        def _is_requirements_missed(target, requirements):
            return list(filter(lambda entry: not target.get(entry), requirements))

        def _check_models_config(config):
            models = config.get('models')
            if not models:
                raise ConfigError('Missed "{}" in local config'.format('models'))

            required_model_entries = ['name', 'launchers', 'datasets']
            required_dataset_entries = ['name']
            required_dataset_error = 'Model {} must specify {} for each dataset'
            for model in models:
                if _is_requirements_missed(model, required_model_entries):
                    raise ConfigError('Each model must specify {}'.format(', '.join(required_model_entries)))

                if list(filter(lambda entry: _is_requirements_missed(entry, required_dataset_entries),
                               model['datasets'])):
                    raise ConfigError(required_dataset_error.format(model['name'], ', '.join(required_dataset_entries)))

        def _check_pipelines_config(config):
            def _count_entry(stages, entry):
                count = 0
                for stage in stages:
                    if entry in stage:
                        count += 1
                return count
            required_pipeline_entries = ['name', 'device_info', 'stages']
            pipelines = config['pipelines']
            if not pipelines:
                raise ConfigError('Missed "{}" in local config'.format('pipelines'))
            for pipeline in pipelines:
                if _is_requirements_missed(pipeline, required_pipeline_entries):
                    raise ConfigError('Each pipeline must specify {}'.format(', '.join(required_pipeline_entries)))
                stages = pipeline['stages']
                first_stage = stages[0]
                dataset = first_stage.get('dataset')
                if not dataset:
                    raise ConfigError('First stage should contain dataset')
                count_datasets = _count_entry(stages, 'dataset')
                if count_datasets != 1:
                    raise ConfigError('Exactly one dataset per pipeline is supported')
                count_launchers = _count_entry(stages, 'launcher')
                if not count_launchers:
                    raise ConfigError('Launchers are not specified')
                count_metrics = _count_entry(stages, 'metrics')
                if not count_metrics:
                    raise ConfigError('Metrics are not specified')

        if 'pipelines' in config:
            _check_pipelines_config(config)
            return 'pipelines'

        _check_models_config(config)
        return 'models'

    @staticmethod
    def _prepare_global_configs(global_configs):
        if not global_configs or 'datasets' not in global_configs:
            return

        datasets = global_configs['datasets']

        def merge(local_entries, global_entries, identifier):
            if not local_entries or not global_entries:
                return

            for i, local in enumerate(local_entries):
                local_identifier = local.get(identifier)
                if not local_identifier:
                    continue

                local_entries[i] = ConfigReader._merge_configs_by_identifier(global_entries, local, identifier)

        for dataset in datasets:
            merge(dataset.get('preprocessing'), global_configs.get('preprocessing'), 'type')
            merge(dataset.get('metrics'), global_configs.get('metrics'), 'type')
            merge(dataset.get('postprocessing'), global_configs.get('postprocessing'), 'type')

    @staticmethod
    def _merge_configs(global_configs, local_config, mode='models'):
        def _merge_models_config(global_configs, local_config):
            config = copy.deepcopy(local_config)
            if not global_configs:
                return config

            models = config.get('models')
            for model in models:
                for i, launcher_entry in enumerate(model['launchers']):
                    model['launchers'][i] = ConfigReader._merge_configs_by_identifier(
                        global_configs['launchers'], launcher_entry, 'framework'
                    )

                for i, dataset in enumerate(model['datasets']):
                    model['datasets'][i] = ConfigReader._merge_configs_by_identifier(
                        global_configs['datasets'], dataset, 'name'
                    )

            return config

        def _merge_pipelines_config(global_config, local_config):
            config = copy.deepcopy(local_config)
            pipelines = []
            raw_pipelines = local_config['pipelines']
            for pipeline in raw_pipelines:
                device_infos = pipeline['device_info']
                per_device_pipelines = []
                for device_info in device_infos:
                    copy_pipeline = copy.deepcopy(pipeline)
                    for stage in copy_pipeline['stages']:
                        if 'launcher' in stage:
                            stage['launcher'].update(device_info)
                    per_device_pipelines.append(copy_pipeline)
                pipelines.extend(per_device_pipelines)
            config['pipelines'] = pipelines

            return config

        functors_by_mode = {
            'models': _merge_models_config,
            'pipelines': _merge_pipelines_config
        }

        return functors_by_mode[mode](global_configs, local_config)

    @staticmethod
    def _merge_configs_by_identifier(global_config, local_config, identifier):
        local_identifier = local_config.get(identifier)
        if local_identifier is None:
            return local_config

        matched = []
        for config in global_config:
            global_identifier = config.get(identifier)
            if global_identifier is None:
                continue

            if global_identifier != local_identifier:
                continue

            matched.append(config)

        config = copy.deepcopy(matched[0] if matched else {})
        for key, value in local_config.items():
            config[key] = value

        return config

    @staticmethod
    def _merge_paths_with_prefixes(arguments, config, mode='models'):
        args = arguments if isinstance(arguments, dict) else vars(arguments)
        entries_paths = {
            'launchers': {
                'model': 'models',
                'weights': 'models',
                'caffe_model': 'models',
                'caffe_weights': 'models',
                'tf_model': 'models',
                'tf_meta': 'models',
                'mxnet_weights': 'models',
                'onnx_model': 'models',
                'kaldi_model': 'models',
                'cpu_extensions': 'extensions',
                'gpu_extensions': 'extensions',
                'bitstream': 'bitstreams',
                'affinity_map' : 'affinity_map'
            },
            'datasets': {
                'segmentation_masks_source': 'source',
                'annotation': 'annotations',
                'dataset_meta': 'annotations',
                'data_source': 'source',
            },
        }

        def merge_entry_paths(keys, value):
            for field, argument in keys.items():
                if field not in value:
                    continue

                config_path = Path(value[field])
                if config_path.is_absolute():
                    value[field] = Path(value[field])
                    continue

                if not argument in args or not args[argument]:
                    continue

                value[field] = args[argument] / config_path

        def create_command_line_for_conversion(config):
            mapping = {}
            value = 'source'
            for key in config:
                if key.endswith('file') or key.endswith('dir'):
                    mapping[key] = value
            return mapping

        def process_config(config_item, entries_paths, dataset_identifier='datasets', identifers_mapping=None):
            for entry, command_line_arg in entries_paths.items():
                entry_id = entry if not identifers_mapping else identifers_mapping[entry]
                if entry_id not in config_item:
                    continue

                if entry_id == dataset_identifier:
                    datasets_configs = config_item[entry_id]
                    if not isinstance(datasets_configs, list):
                        datasets_configs = [datasets_configs]
                    for datasets_config in datasets_configs:
                        annotation_conversion_config = datasets_config.get('annotation_conversion')
                        if annotation_conversion_config:
                            command_line_conversion = (create_command_line_for_conversion(annotation_conversion_config))
                            merge_entry_paths(command_line_conversion, annotation_conversion_config)

                config_entires = config_item[entry_id]
                if not isinstance(config_entires, list):
                    config_entires = [config_entires]
                for config_entry in config_entires:
                    merge_entry_paths(command_line_arg, config_entry)

        def process_models(config, entries_paths):
            for model in config['models']:
                process_config(model, entries_paths)

        def process_pipelines(config, entries_paths):
            identifiers_mapping = {'datasets': 'dataset', 'launchers': 'launcher', 'reader': 'reader'}
            entries_paths.update({'reader': {'data_source': 'source'}})
            for pipeline in config['pipelines']:
                for stage in pipeline['stages']:
                    process_config(stage, entries_paths, 'dataset', identifiers_mapping)

        functors_by_mode = {
            'models': process_models,
            'pipelines': process_pipelines
        }

        processing_func = functors_by_mode[mode]
        processing_func(config, entries_paths)

    @staticmethod
    def _provide_cmd_arguments(arguments, config, mode):
        def merge_converted_model_path(converted_models_dir, mo_output_dir):
            if mo_output_dir:
                mo_output_dir = Path(mo_output_dir)
                if mo_output_dir.is_absolute():
                    return mo_output_dir
                return converted_models_dir / mo_output_dir
            return converted_models_dir

        def merge_dlsdk_launcher_args(arguments, launcher_entry, update_launcher_entry):
            if launcher_entry['framework'].lower() != 'dlsdk':
                return launcher_entry

            launcher_entry.update(update_launcher_entry)
            models_prefix = arguments.models
            if models_prefix:
                launcher_entry['_models_prefix'] = models_prefix

            if not arguments.converted_models:
                return launcher_entry

            mo_params = launcher_entry.get('mo_params', {})

            mo_params.update({
                'output_dir': merge_converted_model_path(arguments.converted_models, mo_params.get('output_dir'))
            })

            launcher_entry['mo_params'] = mo_params

            if arguments.aocl:
                launcher_entry['_aocl'] = arguments.aocl

            return launcher_entry

        def merge_models(config, arguments, update_launcher_entry):
            for model in config['models']:
                for launcher_entry in model['launchers']:
                    merge_dlsdk_launcher_args(arguments, launcher_entry, update_launcher_entry)

        def merge_pipelines(config, arguments, update_launcher_entry):
            for pipeline in config['pipelines']:
                for stage in pipeline['stages']:
                    if 'launcher' in stage:
                        merge_dlsdk_launcher_args(arguments, stage['launcher'], update_launcher_entry)
        functors_by_mode = {
            'models': merge_models,
            'pipelines': merge_pipelines
        }

        additional_keys = [
            'model_optimizer', 'tf_custom_op_config_dir',
            'tf_obj_detection_api_pipeline_config_path',
            'cpu_extensions_mode', 'vpu_log_level'
        ]
        arguments_dict = arguments if isinstance(arguments, dict) else vars(arguments)
        update_launcher_entry = {}

        for key in additional_keys:
            value = arguments_dict.get(key)
            if value:
                update_launcher_entry['_{}'.format(key)] = value

        return functors_by_mode[mode](config, arguments, update_launcher_entry)

    @staticmethod
    def _filter_launchers(config, arguments, mode='models'):
        def filtered(launcher, targets):
            target_tags = args.get('target_tags') or []
            if target_tags:
                if not contains_any(target_tags, launcher.get('tags', [])):
                    return True

            config_framework = launcher['framework'].lower()
            target_framework = (args.get('target_framework') or config_framework).lower()
            if config_framework != target_framework:
                return True

            return targets and launcher.get('device', '').lower() not in targets

        def filter_models(config, target_devices):
            for model in config['models']:
                launchers = model['launchers']
                launchers = [launcher for launcher in launchers if not filtered(launcher, target_devices)]

                if not launchers:
                    warnings.warn('Model "{}" has no launchers'.format(model['name']))

                model['launchers'] = launchers

        def filter_pipelines(config, target_devices):
            saved_pipelines = []
            for pipeline in config['pipelines']:
                filtered_pipeline = False
                for stage in pipeline:
                    if 'launcher' in stage:
                        if filtered(stage['launcher'], target_devices):
                            filtered_pipeline = True
                        break
                if filtered_pipeline:
                    continue
                saved_pipelines.append(pipeline)
            config['pipelines'] = saved_pipelines

        functors_by_mode = {
            'models': filter_models,
            'pipelines': filter_pipelines
        }

        args = arguments if isinstance(arguments, dict) else vars(arguments)
        target_devices = to_lower_register(args.get('target_devices') or [])
        filtering_mode = functors_by_mode[mode]
        filtering_mode(config, target_devices)
