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
from argparse import Namespace

import pytest
from accuracy_checker.config import ConfigReader, ConfigError


class TestConfigReader:
    def setup_method(self):
        self.global_launchers = [
            {
                'framework': 'dlsdk',
                'device': 'fpga',
                'cpu_extensions': 'dlsdk_shared.so',
                'bitstream': 'bitstream'
            },
            {
                'framework': 'caffe',
                'device': 'gpu_0'
            }
        ]

        self.global_datasets = [
            {
                'name': 'global_dataset',
                'annotation': Path('/pascal_voc_2007_annotation.pickle'),
                'data_source': Path('/VOCdevkit/VOC2007/JPEGImages'),
                'preprocessing': [
                    {
                        'type': 'resize',
                        'interpolation': 'mean_image',
                    },
                    {
                        'type': 'normalization',
                        'mean': 'voc',
                    }
                ],
                'metrics': [{
                    'type': 'fppi',
                    'mr_rates': [0.0, 0.1]
                }],
                'postprocessing': [
                    {
                        'type': 'filter',
                        'labels': ['dog', 'airplane'],
                        'min_confidence': 0.05,
                        'min_box_size': 60,
                    },
                    {
                        'type': 'nms',
                        'overlap': 0.5
                    }
                ]
            }
        ]

        self.global_config = {
            'launchers': self.global_launchers,
            'datasets': self.global_datasets
        }

        self.module = 'accuracy_checker.config.ConfigReader'
        self.arguments = Namespace(**{
            'models': Path('models'),
            'extensions': Path('extensions'),
            'source': Path('source'),
            'annotations': Path('annotations'),
            'converted_models': Path('converted_models'),
            'model_optimizer': Path('model_optimizer'),
            'bitstreams': Path('bitstreams'),
            'definitions': None,
            'stored_predictions': None,
            'tf_custom_op_config': None,
            'tf_obj_detection_api_pipeline_config_path': None,
            'progress': 'bar',
            'target_framework': None,
            'target_devices': None,
            'log_file': None,
            'target_tags': None,
            'cpu_extensions_mode': None,
            'aocl': None
        })

    def test_read_configs_without_global_config(self, mocker):
        config = {'models': [{
            'name': 'model',
            'launchers': [{'framework': 'dlsdk', 'model': Path('/absolute_path'), 'weights': Path('/absolute_path')}],
            'datasets': [{'name': 'global_dataset'}]
        }]}
        empty_args = Namespace(**{
            'models': None, 'extensions': None, 'source': None, 'annotations': None,
            'converted_models': None, 'model_optimizer': None, 'bitstreams': None,
            'definitions': None, 'config': None, 'stored_predictions': None, 'tf_custom_op_config': None,
            'progress': 'bar', 'target_framework': None, 'target_devices': None, 'log_file': None,
            'tf_obj_detection_api_pipeline_config_path': None, 'target_tags': None, 'cpu_extensions_mode': None,
            'aocl': None
        })
        mocker.patch('accuracy_checker.utils.get_path', return_value=Path.cwd())
        mocker.patch('yaml.load', return_value=config)
        mocker.patch('pathlib.Path.open')

        result = ConfigReader.merge(empty_args)

        assert 'models' == result[1]
        assert config == result[0]

    def test_empty_local_config_raises_value_error_exception(self, mocker):
        mocker.patch(self.module + '._read_configs', return_value=(
            self.global_config, {}
        ))

        with pytest.raises(ConfigError) as exception:
            ConfigReader.merge(self.arguments)

        error_message = str(exception).split(sep=': ')[-1]
        assert error_message == 'Missing local config'

    def test_missed_models_in_local_config_raises_value_error_exception(self, mocker):
        mocker.patch(self.module + '._read_configs', return_value=(
            self.global_config, {'not_models': 'custom'}
        ))

        with pytest.raises(ConfigError) as exception:
            ConfigReader.merge(self.arguments)

        error_message = str(exception).split(sep=': ')[-1]
        assert error_message == 'Missed "{}" in local config'.format('models')

    def test_empty_models_in_local_config_raises_value_error_exception(self, mocker):
        mocker.patch(self.module + '._read_configs', return_value=(
            self.global_config, {'models': []}
        ))

        with pytest.raises(ConfigError) as exception:
            ConfigReader.merge(self.arguments)

        error_message = str(exception).split(sep=': ')[-1]
        assert error_message == 'Missed "{}" in local config'.format('models')

    def test_missed_name_in_model_raises_value_error_exception(self, mocker):
        mocker.patch(self.module + '._read_configs', return_value=(
            self.global_config, {'models': [{'launchers': None, 'datasets': None}]}
        ))

        with pytest.raises(ConfigError) as exception:
            ConfigReader.merge(self.arguments)

        error_message = str(exception).split(sep=': ')[-1]
        assert error_message == 'Each model must specify {}'.format(', '.join(['name', 'launchers', 'datasets']))

    def test_missed_launchers_in_model_raises_value_error_exception(self, mocker):
        mocker.patch(self.module + '._read_configs', return_value=(
            self.global_config, {'models': [{'name': None, 'datasets': None}]}
        ))

        with pytest.raises(ConfigError) as exception:
            ConfigReader.merge(self.arguments)

        error_message = str(exception).split(sep=': ')[-1]
        assert error_message == 'Each model must specify {}'.format(', '.join(['name', 'launchers', 'datasets']))

    def test_missed_datasets_in_model_raises_value_error_exception(self, mocker):
        mocker.patch(self.module + '._read_configs', return_value=(
            self.global_config, {'models': [{'name': None, 'launchers': None}]}
        ))

        with pytest.raises(ConfigError) as exception:
            ConfigReader.merge(self.arguments)

        error_message = str(exception).split(sep=': ')[-1]
        assert error_message == 'Each model must specify {}'.format(', '.join(['name', 'launchers', 'datasets']))

    def test_invalid_model_raises_value_error_exception(self, mocker):
        mocker.patch(self.module + '._read_configs', return_value=(
            self.global_config, {'models': [{'name': None, 'launchers': None, 'datasets': None}]}
        ))

        with pytest.raises(ConfigError) as exception:
            ConfigReader.merge(self.arguments)

        error_message = str(exception).split(sep=': ')[-1]
        assert error_message == 'Each model must specify {}'.format(', '.join(['name', 'launchers', 'datasets']))

    def test_empty_pipeline_in_local_config_raises_value_error_exception(self, mocker):
        mocker.patch(self.module + '._read_configs', return_value=(
            self.global_config, {'pipelines': []}
        ))

        with pytest.raises(ConfigError) as exception:
            ConfigReader.merge(self.arguments)

        error_message = str(exception).split(sep=': ')[-1]
        assert error_message == 'Missed "{}" in local config'.format('pipelines')

    def test_missed_name_in_pipeline_raises_value_error_exception(self, mocker):
        mocker.patch(self.module + '._read_configs', return_value=(
            self.global_config, {'pipelines': [{'device_info': None, 'stages': None}]}
        ))

        with pytest.raises(ConfigError) as exception:
            ConfigReader.merge(self.arguments)

        error_message = str(exception).split(sep=': ')[-1]
        assert error_message == 'Each pipeline must specify {}'.format(', '.join(['name', 'device_info', 'stages']))

    def test_missed_device_info_in_pipeline_raises_value_error_exception(self, mocker):
        mocker.patch(self.module + '._read_configs', return_value=(
            self.global_config, {'pipelines': [{'name': None, 'stages': None}]}
        ))

        with pytest.raises(ConfigError) as exception:
            ConfigReader.merge(self.arguments)

        error_message = str(exception).split(sep=': ')[-1]
        assert error_message == 'Each pipeline must specify {}'.format(', '.join(['name', 'device_info', 'stages']))

    def test_missed_stages_in_pipeline_raises_value_error_exception(self, mocker):
        mocker.patch(self.module + '._read_configs', return_value=(
            self.global_config, {'pipelines': [{'name': None, 'device_info': None}]}
        ))

        with pytest.raises(ConfigError) as exception:
            ConfigReader.merge(self.arguments)

        error_message = str(exception).split(sep=': ')[-1]
        assert error_message == 'Each pipeline must specify {}'.format(', '.join(['name', 'device_info', 'stages']))

    def test_invalid_pipeline_raises_value_error_exception(self, mocker):
        mocker.patch(self.module + '._read_configs', return_value=(
            self.global_config, {'pipelines': [{'name': None, 'device_info': None, 'stages': None}]}
        ))

        with pytest.raises(ConfigError) as exception:
            ConfigReader.merge(self.arguments)

        error_message = str(exception).split(sep=': ')[-1]
        assert error_message == 'Each pipeline must specify {}'.format(', '.join(['name', 'device_info', 'stages']))

    def test_pipeline_empty_stages_raises_value_error_exception(self, mocker):
        mocker.patch(self.module + '._read_configs', return_value=(
            self.global_config, {'pipelines': [{'name': 'stage1', 'device_info': [{'framework': 'caffe', 'device': 'CPU'}], 'stages': []}]}
        ))

        with pytest.raises(ConfigError) as exception:
            ConfigReader.merge(self.arguments)

        error_message = str(exception).split(sep=': ')[-1]
        assert error_message == 'Each pipeline must specify {}'.format(', '.join(['name', 'device_info', 'stages']))

    def test_pipeline_empty_device_info_raises_value_error_exception(self, mocker):
        mocker.patch(self.module + '._read_configs', return_value=(
            self.global_config, {'pipelines': [{'name': 'stage1', 'device_info': [], 'stages': [{'stage1': {}}]}]}
        ))

        with pytest.raises(ConfigError) as exception:
            ConfigReader.merge(self.arguments)

        error_message = str(exception).split(sep=': ')[-1]
        assert error_message == 'Each pipeline must specify {}'.format(', '.join(['name', 'device_info', 'stages']))

    def test_pipeline_stage_does_not_contain_dataset_raises_value_error_exception(self, mocker):
        mocker.patch(self.module + '._read_configs', return_value=(
            self.global_config, {
                'pipelines': [{'name': 'stage1', 'device_info': [{'framework': 'caffe', 'device': 'CPU'}],
                               'stages': [{'stage': 'stage1'}]}]}
        ))

        with pytest.raises(ConfigError) as exception:
            ConfigReader.merge(self.arguments)

        error_message = str(exception).split(sep=': ')[-1]
        assert error_message == 'First stage should contain dataset'

    def test_pipeline_contains_several_datasets_raises_value_error_exception(self, mocker):
        dataset_config = {
            'name': 'global_dataset',
            'dataset_meta': 'relative_annotation_path',
            'data_source': 'relative_source_path',
            'segmentation_masks_source': 'relative_source_path',
            'annotation': 'relative_annotation_path'
        }
        launcher_config = {'framework': 'dlsdk', 'model': '/absolute_path', 'weights': '/absolute_path'}
        pipelines_config = [
            {'name': 'pipeline', 'device_info': [{'framework': 'caffe', 'device': 'CPU'}],
             'stages': [{'stage': 'stage1', 'dataset': dataset_config},
                        {'stage': 'stage2', 'dataset': dataset_config, 'launcher': launcher_config, 'metrics': {}}
                        ]
             }
        ]
        mocker.patch(self.module + '._read_configs', return_value=(
            self.global_config, {
                'pipelines': pipelines_config}
        ))

        with pytest.raises(ConfigError) as exception:
            ConfigReader.merge(self.arguments)

        error_message = str(exception).split(sep=': ')[-1]
        assert error_message == 'Exactly one dataset per pipeline is supported'

    def test_pipeline_without_launchers_raises_value_error_exception(self, mocker):
        dataset_config = {
                'name': 'global_dataset',
                'dataset_meta': 'relative_annotation_path',
                'data_source': 'relative_source_path',
                'segmentation_masks_source': 'relative_source_path',
                'annotation': 'relative_annotation_path'
            }
        mocker.patch(self.module + '._read_configs', return_value=(
            self.global_config, {
                'pipelines': [{'name': 'stage1', 'device_info': [{'framework': 'caffe', 'device': 'CPU'}],
                               'stages': [{'stage': 'stage1', 'dataset': dataset_config}]}]}
        ))

        with pytest.raises(ConfigError) as exception:
            ConfigReader.merge(self.arguments)

        error_message = str(exception).split(sep=': ')[-1]
        assert error_message == 'Launchers are not specified'

    def test_pipeline_without_metrics_raises_value_error_exception(self, mocker):
        dataset_config = {
                'name': 'global_dataset',
                'dataset_meta': 'relative_annotation_path',
                'annotation': 'relative_annotation_path'
            }
        launcher_config = {'framework': 'dlsdk', 'model': '/absolute_path', 'weights': '/absolute_path'}
        mocker.patch(self.module + '._read_configs', return_value=(
            None, {
                'pipelines': [{'name': 'stage1', 'device_info': [{'framework': 'caffe', 'device': 'CPU'}],
                               'stages': [{'stage': 'stage1', 'dataset': dataset_config, 'launcher': launcher_config}]}]}
        ))

        with pytest.raises(ConfigError) as exception:
            ConfigReader.merge(self.arguments)

        error_message = str(exception).split(sep=': ')[-1]
        assert error_message == 'Metrics are not specified'

    def test_merge_datasets_with_definitions(self, mocker):
        local_config = {'models': [{
            'name': 'model',
            'launchers': [{'framework': 'dlsdk', 'model': '/absolute_path', 'weights': '/absolute_path'}],
            'datasets': [{'name': 'global_dataset'}]
        }]}
        mocker.patch(self.module + '._read_configs', return_value=(
            self.global_config, local_config
        ))
        arguments = copy.deepcopy(self.arguments)
        arguments.model_optimizer = None

        config = ConfigReader.merge(arguments)[0]

        assert config['models'][0]['datasets'][0] == self.global_datasets[0]

    def test_merge_datasets_with_definitions_and_meta_is_not_modified(self, mocker):
        local_config = {'models': [{
            'name': 'model',
            'launchers': [{'framework': 'dlsdk', 'model': '/absolute_path', 'weights': '/absolute_path'}],
            'datasets': [{'name': 'global_dataset', 'dataset_meta': '/absolute_path'}]
        }]}
        expected = self.global_datasets[0]
        expected['dataset_meta'] = Path('/absolute_path')
        mocker.patch(self.module + '._read_configs', return_value=(
            self.global_config, local_config
        ))

        config = ConfigReader.merge(self.arguments)[0]

        assert config['models'][0]['datasets'][0] == expected

    def test_expand_relative_paths_in_datasets_config_using_command_line(self, mocker):
        local_config = {'models': [{
            'name': 'model',
            'launchers': [{'framework': 'caffe'}],
            'datasets': [{
                'name': 'global_dataset',
                'dataset_meta': 'relative_annotation_path',
                'data_source': 'relative_source_path',
                'segmentation_masks_source': 'relative_source_path',
                'annotation': 'relative_annotation_path'
            }]
        }]}

        mocker.patch(self.module + '._read_configs', return_value=(
            None, local_config
        ))
        expected = copy.deepcopy(local_config['models'][0]['datasets'][0])
        expected['annotation'] = self.arguments.annotations / 'relative_annotation_path'
        expected['dataset_meta'] = self.arguments.annotations / 'relative_annotation_path'
        expected['segmentation_masks_source'] = self.arguments.source / 'relative_source_path'
        expected['data_source'] = self.arguments.source / 'relative_source_path'

        config = ConfigReader.merge(self.arguments)[0]

        assert config['models'][0]['datasets'][0] == expected

    def test_not_modify_absolute_paths_in_datasets_config_using_command_line(self):
        local_config = {'models': [{
            'name': 'model',
            'datasets': [{
                'name': 'global_dataset',
                'dataset_meta': '/absolute_annotation_meta_path',
                'data_source': '/absolute_source_path',
                'annotation': '/absolute_annotation_path',
            }]
        }]}

        expected = copy.deepcopy(local_config['models'][0]['datasets'][0])
        expected['annotation'] = Path('/absolute_annotation_path')
        expected['dataset_meta'] = Path('/absolute_annotation_meta_path')
        expected['data_source'] = Path('/absolute_source_path')

        ConfigReader._merge_paths_with_prefixes(self.arguments, local_config)

        assert local_config['models'][0]['datasets'][0] == expected

    def test_expand_relative_paths_in_pipeline_stage_dataset_config_using_command_line(self, mocker):
        dataset_config = {
                'name': 'global_dataset',
                'dataset_meta': 'relative_annotation_path',
                'data_source': 'relative_source_path',
                'segmentation_masks_source': 'relative_source_path',
                'annotation': 'relative_annotation_path'
            }
        launcher_config = {'framework': 'dlsdk', 'model': '/absolute_path', 'weights': '/absolute_path'}
        pipelines_config = [
            {
                'name': 'pipeline', 'device_info': [{'framework': 'caffe', 'device': 'CPU'}],
                'stages': [
                    {'stage': 'stage1', 'dataset': dataset_config},
                    {'stage': 'stage2', 'launcher': launcher_config, 'metrics': {}}
                ]
            }
        ]
        mocker.patch(self.module + '._read_configs', return_value=(
            None, {
                'pipelines': pipelines_config}
        ))

        expected = copy.deepcopy(dataset_config)
        expected['annotation'] = self.arguments.annotations / 'relative_annotation_path'
        expected['dataset_meta'] = self.arguments.annotations / 'relative_annotation_path'
        expected['segmentation_masks_source'] = self.arguments.source / 'relative_source_path'
        expected['data_source'] = self.arguments.source / 'relative_source_path'

        config = ConfigReader.merge(self.arguments)[0]

        assert config['pipelines'][0]['stages'][0]['dataset'] == expected

    def test_not_modify_absolute_paths_in_pipeline_stage_dataset_config_using_command_line(self, mocker):
        dataset_config = {
            'name': 'global_dataset',
            'dataset_meta': '/absolute_annotation_meta_path',
            'data_source': '/absolute_source_path',
            'annotation': '/absolute_annotation_path'
        }
        launcher_config = {'framework': 'dlsdk', 'model': '/absolute_path', 'weights': '/absolute_path'}
        pipelines_config = [
            {
                'name': 'pipeline', 'device_info': [{'device': 'CPU'}],
                'stages': [
                    {'stage': 'stage1', 'dataset': dataset_config},
                    {'stage': 'stage2', 'launcher': launcher_config, 'metrics': {}}
                ]
            }
        ]
        mocker.patch(self.module + '._read_configs', return_value=(
            None, {
                'pipelines': pipelines_config}
        ))

        expected = copy.deepcopy(dataset_config)
        expected['annotation'] = Path('/absolute_annotation_path')
        expected['dataset_meta'] = Path('/absolute_annotation_meta_path')
        expected['data_source'] = Path('/absolute_source_path')

        config = ConfigReader.merge(self.arguments)[0]

        assert config['pipelines'][0]['stages'][0]['dataset'] == expected

    def test_merge_launcher_with_device_info(self, mocker):
        dataset_config = {
            'name': 'global_dataset',
            'dataset_meta': '/absolute_annotation_meta_path',
            'data_source': '/absolute_source_path',
            'annotation': '/absolute_annotation_path'
        }
        launcher_config = {'framework': 'caffe', 'model': Path('/absolute_path'), 'weights': Path('/absolute_path')}
        device_info = {'device': 'CPU'}
        expected = copy.deepcopy(launcher_config)
        expected.update(device_info)
        pipelines_config = [
            {
                'name': 'pipeline', 'device_info': [device_info],
                'stages': [
                    {'stage': 'stage1', 'dataset': dataset_config},
                    {'stage': 'stage2', 'launcher': launcher_config, 'metrics': {}}
                ]
            }
        ]
        mocker.patch(self.module + '._read_configs', return_value=(
            None, {
                'pipelines': pipelines_config}
        ))

        config = ConfigReader.merge(self.arguments)[0]

        assert config['pipelines'][0]['stages'][1]['launcher'] == expected

    def test_merge_launcher_with_2_device_info(self, mocker):
        dataset_config = {
            'name': 'global_dataset',
            'dataset_meta': '/absolute_annotation_meta_path',
            'data_source': '/absolute_source_path',
            'annotation': '/absolute_annotation_path'
        }
        launcher_config = {'framework': 'caffe', 'model': Path('/absolute_path'), 'weights': Path('/absolute_path')}
        device_info = [{'device': 'CPU'}, {'device': 'GPU'}]
        expected = [copy.deepcopy(launcher_config), copy.deepcopy(launcher_config)]
        expected[0].update(device_info[0])
        expected[1].update(device_info[1])
        pipelines_config = [
            {
                'name': 'pipeline', 'device_info': device_info,
                'stages': [
                    {'stage': 'stage1', 'dataset': dataset_config},
                    {'stage': 'stage2', 'launcher': launcher_config, 'metrics': {}}
                ]
            }
        ]
        mocker.patch(self.module + '._read_configs', return_value=(
            None, {
                'pipelines': pipelines_config}
        ))

        config = ConfigReader.merge(self.arguments)[0]
        assert len(config['pipelines']) == 2
        assert config['pipelines'][0]['stages'][1]['launcher'] == expected[0]
        assert config['pipelines'][1]['stages'][1]['launcher'] == expected[1]

    def test_merge_launchers_with_definitions(self, mocker):
        local_config = {'models': [{
            'name': 'model',
            'launchers': [{'framework': 'dlsdk'}],
            'datasets': [{'name': 'global_dataset'}]
        }]}
        mocker.patch(self.module + '._read_configs', return_value=(
            self.global_config, local_config
        ))
        expected = copy.deepcopy(self.get_global_launcher('dlsdk'))
        expected['bitstream'] = self.arguments.bitstreams / expected['bitstream']
        expected['cpu_extensions'] = self.arguments.extensions / expected['cpu_extensions']
        args = copy.deepcopy(self.arguments)
        args.model_optimizer = None
        args.converted_models = None
        args.models = None

        config = ConfigReader.merge(args)[0]

        assert config['models'][0]['launchers'][0] == expected

    def test_merge_launchers_with_model_is_not_modified(self, mocker):
        local_config = {'models': [{
            'name': 'model',
            'launchers': [{'framework': 'dlsdk', 'model': 'custom'}],
            'datasets': [{'name': 'global_dataset'}]
        }]}
        expected = copy.deepcopy(self.get_global_launcher('dlsdk'))
        expected['model'] = 'custom'
        expected['bitstream'] = self.arguments.bitstreams / expected['bitstream']
        expected['cpu_extensions'] = self.arguments.extensions / expected['cpu_extensions']
        mocker.patch(self.module + '._read_configs', return_value=(
            self.global_config, local_config
        ))
        args = copy.deepcopy(self.arguments)
        args.model_optimizer = None
        args.models = None
        args.converted_models = None
        config = ConfigReader.merge(args)[0]

        assert config['models'][0]['launchers'][0] == expected

    def test_expand_relative_paths_in_launchers_config_using_command_line(self, mocker):
        local_config = {'models': [{
            'name': 'model',
            'launchers': [{
                'framework': 'dlsdk',
                'model': 'relative_model_path',
                'weights': 'relative_weights_path',
                'cpu_extensions': 'relative_extensions_path',
                'gpu_extensions': 'relative_extensions_path',
                'caffe_model': 'relative_model_path',
                'caffe_weights': 'relative_weights_path',
                'tf_model': 'relative_model_path',
                'mxnet_weights': 'relative_weights_path',
                'bitstream': 'relative_bitstreams_path'
            }],
            'datasets': [{'name': 'dataset'}]
        }]}
        mocker.patch(self.module + '._read_configs', return_value=(None, local_config))

        expected = copy.deepcopy(local_config['models'][0]['launchers'][0])
        expected['model'] = self.arguments.models / 'relative_model_path'
        expected['caffe_model'] = self.arguments.models / 'relative_model_path'
        expected['tf_model'] = self.arguments.models / 'relative_model_path'
        expected['weights'] = self.arguments.models / 'relative_weights_path'
        expected['caffe_weights'] = self.arguments.models / 'relative_weights_path'
        expected['mxnet_weights'] = self.arguments.models / 'relative_weights_path'
        expected['cpu_extensions'] = self.arguments.extensions / 'relative_extensions_path'
        expected['gpu_extensions'] = self.arguments.extensions / 'relative_extensions_path'
        expected['bitstream'] = self.arguments.bitstreams / 'relative_bitstreams_path'
        expected['_models_prefix'] = self.arguments.models
        args = copy.deepcopy(self.arguments)
        args.model_optimizer = None
        args.converted_models = None
        config = ConfigReader.merge(args)[0]

        assert config['models'][0]['launchers'][0] == expected

    def test_both_launchers_are_filtered_by_target_tags_if_tags_not_provided_in_config(self, mocker):
        config_launchers = [
            {
                'framework': 'dlsdk',
                'model': '/absolute_path1',
                'weights': '/absolute_path1',
                'adapter': 'classification',
                'device': 'CPU',
            },
            {
                'framework': 'dlsdk',
                'model': '/absolute_path2',
                'weights': '/absolute_path2',
                'adapter': 'classification',
                'device': 'GPU',
            }
        ]
        local_config = {'models': [{'name': 'name', 'launchers': config_launchers, 'datasets': [{'name': 'dataset'}]}]}
        self.arguments.target_tags = ['some_tag']

        mocker.patch(self.module + '._read_configs', return_value=(None, local_config))

        with pytest.warns(Warning):
            config = ConfigReader.merge(self.arguments)[0]

        launchers = config['models'][0]['launchers']
        assert len(launchers) == 0

    def test_launcher_is_not_filtered_by_the_same_tag(self, mocker):
        config_launchers = [{
            'framework': 'dlsdk',
            'tags': ['some_tag'],
            'model': Path('/absolute_path1'),
            'weights': Path('/absolute_path1'),
            'adapter': 'classification',
            'device': 'CPU',
            '_model_optimizer': self.arguments.model_optimizer,
            '_models_prefix': self.arguments.models
        }]
        local_config = {'models': [{'name': 'name', 'launchers': config_launchers, 'datasets': [{'name': 'dataset'}]}]}
        mocker.patch(self.module + '._read_configs', return_value=(None, local_config))
        args = copy.deepcopy(self.arguments)
        args.model_optimizer = None
        args.converted_models = None
        args.target_tags = ['some_tag']

        config = ConfigReader.merge(args)[0]

        launchers = config['models'][0]['launchers']
        assert launchers[0] == config_launchers[0]

    def test_both_launchers_are_not_filtered_by_the_same_tag(self, mocker):
        config_launchers = [
            {
                'framework': 'dlsdk',
                'tags': ['some_tag'],
                'model': Path('/absolute_path1'),
                'weights': Path('/absolute_path1'),
                'adapter': 'classification',
                'device': 'CPU',
                '_model_optimizer': self.arguments.model_optimizer,
                '_models_prefix': self.arguments.models
            },
            {
                'framework': 'dlsdk',
                'tags': ['some_tag'],
                'model': Path('/absolute_path2'),
                'weights': Path('/absolute_path2'),
                'adapter': 'classification',
                'device': 'GPU',
                '_model_optimizer': self.arguments.model_optimizer,
                '_models_prefix': self.arguments.models
            }
        ]
        local_config = {'models': [{'name': 'name', 'launchers': config_launchers, 'datasets': [{'name': 'dataset'}]}]}
        mocker.patch(self.module + '._read_configs', return_value=(None, local_config))
        args = copy.deepcopy(self.arguments)
        args.model_optimizer = None
        args.converted_models = None
        args.target_tags = ['some_tag']

        config = ConfigReader.merge(args)[0]

        launchers = config['models'][0]['launchers']
        assert launchers == config_launchers

    def test_both_launchers_are_filtered_by_another_tag(self, mocker):
        config_launchers = [
            {
                'framework': 'dlsdk',
                'tags': ['some_tag'],
                'model': '/absolute_path1',
                'weights': '/absolute_path1',
                'adapter': 'classification',
                'device': 'CPU',
                '_model_optimizer': self.arguments.model_optimizer,
                '_models_prefix': self.arguments.models
            },
            {
                'framework': 'dlsdk',
                'tags': ['some_tag'],
                'model': '/absolute_path2',
                'weights': '/absolute_path2',
                'adapter': 'classification',
                'device': 'GPU',
                '_model_optimizer': self.arguments.model_optimizer,
                '_models_prefix': self.arguments.models
            }
        ]
        local_config = {'models': [{'name': 'name', 'launchers': config_launchers, 'datasets': [{'name': 'dataset'}]}]}
        mocker.patch(self.module + '._read_configs', return_value=(None, local_config))
        args = copy.deepcopy(self.arguments)
        args.model_optimizer = None
        args.converted_models = None
        args.target_tags = ['other_tag']

        with pytest.warns(Warning):
            config = ConfigReader.merge(args)[0]

        launchers = config['models'][0]['launchers']
        assert len(launchers) == 0

    def test_only_appropriate_launcher_is_filtered_by_another_tag(self, mocker):
        config_launchers = [
            {
                'framework': 'dlsdk',
                'tags': ['tag1'],
                'model': Path('/absolute_path1'),
                'weights': Path('/absolute_path1'),
                'adapter': 'classification',
                'device': 'CPU',
                '_model_optimizer': self.arguments.model_optimizer,
                '_models_prefix': self.arguments.models
            },
            {
                'framework': 'caffe',
                'tags': ['tag2'],
                'model': Path('/absolute_path2'),
                'weights': Path('/absolute_path2'),
                'adapter': 'classification',
                'device': 'GPU',
                '_model_optimizer': self.arguments.model_optimizer,
                '_models_prefix': self.arguments.models
            }
        ]
        local_config = {'models': [{'name': 'name', 'launchers': config_launchers, 'datasets': [{'name': 'dataset'}]}]}
        mocker.patch(self.module + '._read_configs', return_value=(None, local_config))
        self.arguments.target_tags = ['tag2']

        config = ConfigReader.merge(self.arguments)[0]

        launchers = config['models'][0]['launchers']
        assert len(launchers) == 1
        assert launchers[0] == config_launchers[1]

    def test_only_appropriate_launcher_is_filtered_by_another_tag_if_provided_several_target_tags(self, mocker):
        config_launchers = [
            {
                'framework': 'dlsdk',
                'tags': ['tag1'],
                'model': Path('/absolute_path1'),
                'weights': Path('/absolute_path1'),
                'adapter': 'classification',
                'device': 'CPU',
                '_model_optimizer': self.arguments.model_optimizer,
                '_models_prefix': self.arguments.models
            },
            {
                'framework': 'caffe',
                'tags': ['tag2'],
                'model': Path('/absolute_path2'),
                'weights': Path('/absolute_path2'),
                'adapter': 'classification',
                'device': 'GPU',
                '_model_optimizer': self.arguments.model_optimizer,
                '_models_prefix': self.arguments.models
            }
        ]
        local_config = {'models': [{'name': 'name', 'launchers': config_launchers, 'datasets': [{'name': 'dataset'}]}]}
        mocker.patch(self.module + '._read_configs', return_value=(None, local_config))
        self.arguments.target_tags = ['tag2', 'tag3']

        config = ConfigReader.merge(self.arguments)[0]

        launchers = config['models'][0]['launchers']
        assert len(launchers) == 1
        assert launchers[0] == config_launchers[1]

    def test_launcher_with_several_tags_contained_at_least_one_from_target_tegs_is_not_filtered(self, mocker):
        config_launchers = [
            {
                'framework': 'dlsdk',
                'tags': ['tag1', 'tag2'],
                'model': Path('/absolute_path1'),
                'weights': Path('/absolute_path1'),
                'adapter': 'classification',
                'device': 'CPU',
                '_model_optimizer': self.arguments.model_optimizer,
                '_models_prefix': self.arguments.models
            }
        ]
        local_config = {'models': [{'name': 'name', 'launchers': config_launchers, 'datasets': [{'name': 'dataset'}]}]}
        mocker.patch(self.module + '._read_configs', return_value=(None, local_config))
        args = copy.deepcopy(self.arguments)
        args.model_optimizer = None
        args.converted_models = None
        args.target_tags = ['tag2']

        config = ConfigReader.merge(args)[0]

        launchers = config['models'][0]['launchers']
        assert len(launchers) == 1
        assert launchers[0] == config_launchers[0]

    def test_both_launchers_with_different_tags_are_not_filtered_by_the_same_tags(self, mocker):
        config_launchers = [
            {
                'framework': 'dlsdk',
                'tags': ['tag1'],
                'model': Path('/absolute_path1'),
                'weights': Path('/absolute_path1'),
                'adapter': 'classification',
                'device': 'CPU',
                '_model_optimizer': self.arguments.model_optimizer,
                '_models_prefix': self.arguments.models
            },
            {
                'framework': 'dlsdk',
                'tags': ['tag2'],
                'model': Path('/absolute_path2'),
                'weights': Path('/absolute_path2'),
                'adapter': 'classification',
                'device': 'GPU',
                '_model_optimizer': self.arguments.model_optimizer,
                '_models_prefix': self.arguments.models
            }
        ]
        local_config = {'models': [{'name': 'name', 'launchers': config_launchers, 'datasets': [{'name': 'dataset'}]}]}
        mocker.patch(self.module + '._read_configs', return_value=(None, local_config))
        args = copy.deepcopy(self.arguments)
        args.model_optimizer = None
        args.converted_models = None
        args.target_tags = ['tag1', 'tag2']

        config = ConfigReader.merge(args)[0]

        launchers = config['models'][0]['launchers']
        assert launchers == config_launchers

    def test_launcher_is_not_filtered_by_the_same_framework(self, mocker):
        config_launchers = [{
            'framework': 'dlsdk',
            'model': Path('/absolute_path1'),
            'weights': Path('/absolute_path1'),
            'adapter': 'classification',
            'device': 'CPU',
            '_model_optimizer': self.arguments.model_optimizer,
            '_models_prefix': self.arguments.models
        }]
        local_config = {'models': [{'name': 'name', 'launchers': config_launchers, 'datasets': [{'name': 'dataset'}]}]}
        mocker.patch(self.module + '._read_configs', return_value=(None, local_config))
        args = copy.deepcopy(self.arguments)
        args.model_optimizer = None
        args.converted_models = None
        args.target_framework = 'dlsdk'

        config = ConfigReader.merge(args)[0]

        launchers = config['models'][0]['launchers']
        assert launchers == config_launchers

    def test_both_launchers_are_not_filtered_by_the_same_framework(self, mocker):
        config_launchers = [
            {
                'framework': 'dlsdk',
                'model': Path('/absolute_path1'),
                'weights': Path('/absolute_path1'),
                'adapter': 'classification',
                'device': 'CPU',
                '_model_optimizer': self.arguments.model_optimizer,
                '_models_prefix': self.arguments.models
            },
            {
                'framework': 'dlsdk',
                'model': Path('/absolute_path2'),
                'weights': Path('/absolute_path2'),
                'adapter': 'classification',
                'device': 'GPU',
                '_model_optimizer': self.arguments.model_optimizer,
                '_models_prefix': self.arguments.models
            }
        ]
        local_config = {'models': [{'name': 'name', 'launchers': config_launchers, 'datasets': [{'name': 'dataset'}]}]}
        mocker.patch(self.module + '._read_configs', return_value=(None, local_config))
        args = copy.deepcopy(self.arguments)
        args.model_optimizer = None
        args.converted_models = None
        args.target_framework = 'dlsdk'

        config = ConfigReader.merge(args)[0]

        launchers = config['models'][0]['launchers']
        assert launchers == config_launchers

    def test_launcher_is_filtered_by_another_framework(self, mocker):
        config_launchers = [{
            'framework': 'dlsdk',
            'model': Path('/absolute_path'),
            'weights': Path('/absolute_path'),
            'adapter': 'classification',
            '_model_optimizer': self.arguments.model_optimizer,
            '_models_prefix': self.arguments.models
        }]
        local_config = {'models': [{'name': 'name', 'launchers': config_launchers, 'datasets': [{'name': 'dataset'}]}]}
        mocker.patch(self.module + '._read_configs', return_value=(None, local_config))
        self.arguments.target_framework = 'caffe'

        with pytest.warns(Warning):
            config = ConfigReader.merge(self.arguments)[0]

        launchers = config['models'][0]['launchers']
        assert len(launchers) == 0

    def test_both_launchers_are_filtered_by_another_framework(self, mocker):
        config_launchers = [
            {
                'framework': 'dlsdk',
                'model': '/absolute_path1',
                'weights': '/absolute_path1',
                'adapter': 'classification',
                'device': 'CPU',
                '_model_optimizer': self.arguments.model_optimizer,
                '_models_prefix': self.arguments.models
            },
            {
                'framework': 'dlsdk',
                'model': '/absolute_path2',
                'weights': '/absolute_path2',
                'adapter': 'classification',
                'device': 'GPU',
                '_model_optimizer': self.arguments.model_optimizer,
                '_models_prefix': self.arguments.models
            }
        ]
        local_config = {'models': [{'name': 'name', 'launchers': config_launchers, 'datasets': [{'name': 'dataset'}]}]}
        mocker.patch(self.module + '._read_configs', return_value=(None, local_config))
        self.arguments.target_framework = 'caffe'

        with pytest.warns(Warning):
            config = ConfigReader.merge(self.arguments)[0]

        launchers = config['models'][0]['launchers']
        assert len(launchers) == 0

    def test_only_appropriate_launcher_is_filtered_by_another_framework(self, mocker):
        config_launchers = [
            {
                'framework': 'dlsdk',
                'model': Path('/absolute_path1'),
                'weights': Path('/absolute_path1'),
                'adapter': 'classification',
                'device': 'CPU',
                '_model_optimizer': self.arguments.model_optimizer,
                '_models_prefix': self.arguments.models
            },
            {
                'framework': 'caffe',
                'model': Path('/absolute_path2'),
                'weights': Path('/absolute_path2'),
                'adapter': 'classification',
                'device': 'GPU'
            }
        ]
        local_config = {'models': [{'name': 'name', 'launchers': config_launchers, 'datasets': [{'name': 'dataset'}]}]}
        mocker.patch(self.module + '._read_configs', return_value=(None, local_config))
        self.arguments.target_framework = 'caffe'

        config = ConfigReader.merge(self.arguments)[0]

        launchers = config['models'][0]['launchers']
        assert len(launchers) == 1
        assert launchers[0] == config_launchers[1]

    def test_launcher_is_not_filtered_by_the_same_device(self, mocker):
        config_launchers = [{
            'framework': 'dlsdk',
            'model': Path('/absolute_path1'),
            'weights': Path('/absolute_path1'),
            'adapter': 'classification',
            'device': 'CPU',
            '_model_optimizer': self.arguments.model_optimizer,
            '_models_prefix': self.arguments.models
        }]
        local_config = {'models': [{'name': 'name', 'launchers': config_launchers, 'datasets': [{'name': 'dataset'}]}]}
        mocker.patch(self.module + '._read_configs', return_value=(None, local_config))
        args = copy.deepcopy(self.arguments)
        args.model_optimizer = None
        args.converted_models = None
        args.target_devices = ['CPU']

        config = ConfigReader.merge(args)[0]

        launchers = config['models'][0]['launchers']
        assert launchers == config_launchers

    def test_both_launchers_are_not_filtered_by_the_same_device(self, mocker):
        config_launchers = [
            {
                'framework': 'dlsdk',
                'model': Path('/absolute_path1'),
                'weights': Path('/absolute_path1'),
                'adapter': 'classification',
                'device': 'CPU',
                '_model_optimizer': self.arguments.model_optimizer,
                '_models_prefix': self.arguments.models
            },
            {
                'framework': 'caffe',
                'model': Path('/absolute_path2'),
                'weights': Path('/absolute_path2'),
                'adapter': 'classification',
                'device': 'CPU'
            }
        ]
        local_config = {'models': [{'name': 'name', 'launchers': config_launchers, 'datasets': [{'name': 'dataset'}]}]}
        mocker.patch(self.module + '._read_configs', return_value=(None, local_config))
        args = copy.deepcopy(self.arguments)
        args.converted_models = None
        args.target_devices = ['CPU']

        config = ConfigReader.merge(args)[0]

        launchers = config['models'][0]['launchers']
        assert launchers == config_launchers

    def test_launcher_is_filtered_by_another_device(self, mocker):
        config_launchers = [{
            'framework': 'dlsdk',
            'model': Path('/absolute_path1'),
            'weights': Path('/absolute_path1'),
            'adapter': 'classification',
            'device': 'CPU',
            '_model_optimizer': self.arguments.model_optimizer,
            '_models_prefix': self.arguments.models
        }]
        local_config = {'models': [{'name': 'name', 'launchers': config_launchers, 'datasets': [{'name': 'dataset'}]}]}
        mocker.patch(self.module + '._read_configs', return_value=(None, local_config))
        args = copy.deepcopy(self.arguments)
        args.converted_models = None
        args.target_devices = ['GPU']

        with pytest.warns(Warning):
            config = ConfigReader.merge(args)[0]

        launchers = config['models'][0]['launchers']
        assert len(launchers) == 0

    def test_both_launchers_are_filtered_by_another_device(self, mocker):
        config_launchers = [
            {
                'framework': 'dlsdk',
                'model': Path('/absolute_path1'),
                'weights': Path('/absolute_path1'),
                'adapter': 'classification',
                'device': 'CPU',
                '_model_optimizer': self.arguments.model_optimizer,
                '_models_prefix': self.arguments.models
            },
            {
                'framework': 'caffe',
                'model': Path('/absolute_path2'),
                'weights': Path('/absolute_path2'),
                'adapter': 'classification',
                'device': 'CPU'
            }
        ]
        local_config = {'models': [{'name': 'name', 'launchers': config_launchers, 'datasets': [{'name': 'dataset'}]}]}
        mocker.patch(self.module + '._read_configs', return_value=(None, local_config))
        self.arguments.target_devices = ['GPU']

        with pytest.warns(Warning):
            config = ConfigReader.merge(self.arguments)[0]

        launchers = config['models'][0]['launchers']
        assert len(launchers) == 0

    def test_only_appropriate_launcher_is_filtered_by_another_device(self, mocker):
        config_launchers = [
            {
                'framework': 'dlsdk',
                'model': Path('/absolute_path1'),
                'weights': Path('/absolute_path1'),
                'adapter': 'classification',
                'device': 'CPU',
                '_model_optimizer': self.arguments.model_optimizer,
                '_models_prefix': self.arguments.models
            },
            {
                'framework': 'caffe',
                'model': Path('/absolute_path2'),
                'weights': Path('/absolute_path2'),
                'adapter': 'classification',
                'device': 'GPU'
            }
        ]
        local_config = {'models': [{'name': 'name', 'launchers': config_launchers, 'datasets': [{'name': 'dataset'}]}]}
        mocker.patch(self.module + '._read_configs', return_value=(None, local_config))
        args = copy.deepcopy(self.arguments)
        args.converted_models = None
        args.target_devices = ['GPU']

        config = ConfigReader.merge(args)[0]

        launchers = config['models'][0]['launchers']
        assert len(launchers) == 1
        assert launchers[0] == config_launchers[1]

    def test_only_appropriate_launcher_is_filtered_by_user_input_devices(self, mocker):
        config_launchers = [
            {
                'framework': 'dlsdk',
                'model': Path('/absolute_path1'),
                'weights': Path('/absolute_path1'),
                'adapter': 'classification',
                'device': 'CPU',
                '_model_optimizer': self.arguments.model_optimizer,
                '_models_prefix': self.arguments.models
            },
            {
                'framework': 'dlsdk',
                'model': Path('/absolute_path1'),
                'weights': Path('/absolute_path1'),
                'adapter': 'classification',
                'device': 'HETERO:CPU,GPU',
                '_model_optimizer': self.arguments.model_optimizer,
                '_models_prefix': self.arguments.models
            },
            {
                'framework': 'caffe',
                'model': Path('/absolute_path2'),
                'weights': Path('/absolute_path2'),
                'adapter': 'classification',
                'device': 'GPU',
            }
        ]

        local_config = {'models': [{'name': 'name', 'launchers': config_launchers, 'datasets': [{'name': 'dataset'}]}]}
        mocker.patch(self.module + '._read_configs', return_value=(None, local_config))
        args = copy.deepcopy(self.arguments)
        args.converted_models = None
        args.target_devices = ['GPU', 'CPU']

        config = ConfigReader.merge(args)[0]

        launchers = config['models'][0]['launchers']
        assert launchers == [config_launchers[0], config_launchers[2]]

    def test_both_launchers_are_filtered_by_other_devices(self, mocker):
        config_launchers = [
            {
                'framework': 'dlsdk',
                'model': '/absolute_path1',
                'weights': '/absolute_path1',
                'adapter': 'classification',
                'device': 'CPU',
            },
            {
                'framework': 'caffe',
                'model': '/absolute_path2',
                'weights': '/absolute_path2',
                'adapter': 'classification',
                'device': 'CPU'
            }
        ]
        local_config = {'models': [{'name': 'name', 'launchers': config_launchers, 'datasets': [{'name': 'dataset'}]}]}
        mocker.patch(self.module + '._read_configs', return_value=(None, local_config))
        self.arguments.target_devices = ['FPGA', 'MYRIAD']

        with pytest.warns(Warning):
            config = ConfigReader.merge(self.arguments)[0]

        launchers = config['models'][0]['launchers']
        assert len(launchers) == 0

    def test_both_launchers_are_not_filtered_by_same_devices(self, mocker):
        config_launchers = [
            {
                'framework': 'dlsdk',
                'model': Path('/absolute_path1'),
                'weights': Path('/absolute_path1'),
                'adapter': 'classification',
                'device': 'CPU',
                '_model_optimizer': self.arguments.model_optimizer,
                '_models_prefix': self.arguments.models
            },
            {
                'framework': 'caffe',
                'model': Path('/absolute_path2'),
                'weights': Path('/absolute_path2'),
                'adapter': 'classification',
                'device': 'GPU'
            }
        ]
        local_config = {'models': [{'name': 'name', 'launchers': config_launchers, 'datasets': [{'name': 'dataset'}]}]}
        mocker.patch(self.module + '._read_configs', return_value=(None, local_config))
        args = copy.deepcopy(self.arguments)
        args.converted_models = None
        args.target_devices = ['GPU', 'CPU']

        config = ConfigReader.merge(args)[0]

        launchers = config['models'][0]['launchers']
        assert launchers == config_launchers

    def test_launcher_is_not_filtered_by_device_with_tail(self, mocker):
        config_launchers = [
            {
                'framework': 'dlsdk',
                'model': Path('/absolute_path1'),
                'weights': Path('/absolute_path1'),
                'adapter': 'classification',
                'device': 'CPU',
                '_model_optimizer': self.arguments.model_optimizer,
                '_models_prefix': self.arguments.models
            },
            {
                'framework': 'caffe',
                'model': Path('/absolute_path2'),
                'weights': Path('/absolute_path2'),
                'adapter': 'classification',
                'device': 'GPU'
            }
        ]
        local_config = {'models': [{'name': 'name', 'launchers': config_launchers, 'datasets': [{'name': 'dataset'}]}]}
        mocker.patch(self.module + '._read_configs', return_value=(None, local_config))
        args = copy.deepcopy(self.arguments)
        args.converted_models = None
        args.target_devices = ['CPU', 'GPU_unexpected_tail']

        config = ConfigReader.merge(args)[0]

        launchers = config['models'][0]['launchers']
        assert len(launchers) == 1
        assert launchers[0] == config_launchers[0]

    def get_global_launcher(self, framework):
        for launcher in self.global_launchers:
            if launcher['framework'] == framework:
                return launcher

        raise ValueError('Undefined global launcher with framework = "{}"'.format(framework))

    def get_global_dataset(self, name):
        for dataset in self.global_datasets:
            if dataset['name'] == name:
                return dataset

        raise ValueError('Undefined global dataset with name = "{}"'.format(name))
