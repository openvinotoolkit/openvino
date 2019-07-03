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
import pytest
from .common import make_representation
from accuracy_checker.config import ConfigError

from accuracy_checker.dataset import Dataset


def copy_dataset_config(config):
    new_config = copy.deepcopy(config)

    return new_config

class MockPreprocessor:
    @staticmethod
    def process(images):
        return images


class TestDataset:
    dataset_config = {
            'name': 'custom',
            'annotation': 'custom',
            'data_source': 'custom',
            'metrics': [{'type': 'map'}]
        }

    def test_missed_name_raises_config_error_exception(self):
        local_dataset = copy_dataset_config(self.dataset_config)
        local_dataset.pop('name')

        with pytest.raises(ConfigError):
            Dataset(local_dataset, MockPreprocessor())

    def test_setting_custom_dataset_with_missed_annotation_raises_config_error_exception(self):
        local_dataset = copy_dataset_config(self.dataset_config)
        local_dataset.pop('annotation')
        with pytest.raises(ConfigError):
            Dataset(local_dataset, MockPreprocessor())

    @pytest.mark.usefixtures('mock_path_exists')
    def test_setting_custom_dataset_with_missed_data_source_raises_config_error_exception(self):
        local_dataset = copy_dataset_config(self.dataset_config)
        local_dataset.pop('data_source')
        with pytest.raises(ConfigError):
            Dataset(local_dataset, MockPreprocessor())


@pytest.mark.usefixtures('mock_path_exists')
class TestAnnotationConversion:
    dataset_config = {
        'name': 'custom',
        'data_source': 'custom',
        'metrics': [{'type': 'map'}]
    }

    def test_annotation_conversion_unknown_converter_raise_config_error(self):
        addition_options = {'annotation_conversion': {'converter': 'unknown'}}
        config = copy_dataset_config(self.dataset_config)
        config.update(addition_options)
        with pytest.raises(ValueError):
            Dataset(config, MockPreprocessor())

    def test_annotation_conversion_converter_without_required_options_raise_config_error(self):
        addition_options = {'annotation_conversion': {'converter': 'wider'}}
        config = copy_dataset_config(self.dataset_config)
        config.update(addition_options)
        with pytest.raises(ConfigError):
            Dataset(config, MockPreprocessor())

    def test_annotation_conversion_raise_config_error_on_extra_args(self):
        addition_options = {'annotation_conversion': {'converter': 'wider', 'annotation_file': 'file', 'something_extra': 'extra'}}
        config = copy_dataset_config(self.dataset_config)
        config.update(addition_options)
        with pytest.raises(ConfigError):
            Dataset(config, MockPreprocessor())

    def test_sucessful_annotation_conversion(self, mocker):
        addition_options = {'annotation_conversion': {'converter': 'wider', 'annotation_file': 'file'}}
        config = copy_dataset_config(self.dataset_config)
        config.update(addition_options)
        annotation_converter_mock = mocker.patch(
            'accuracy_checker.annotation_converters.WiderFormatConverter.convert',
            return_value=(make_representation("0 0 0 5 5", True), None)
        )
        Dataset(config, MockPreprocessor())
        annotation_converter_mock.assert_called_once_with()

    def test_annotation_conversion_with_store_annotation(self, mocker):
        addition_options = {
            'annotation_conversion': {'converter': 'wider', 'annotation_file': 'file'},
            'annotation': 'custom'
        }
        config = copy_dataset_config(self.dataset_config)
        config.update(addition_options)
        converted_annotation = make_representation('0 0 0 5 5', True)
        mocker.patch(
            'accuracy_checker.annotation_converters.WiderFormatConverter.convert',
            return_value=(converted_annotation, None)
        )
        annotation_saver_mock = mocker.patch(
            'accuracy_checker.dataset.save_annotation'
        )
        Dataset(config, MockPreprocessor())

        annotation_saver_mock.assert_called_once_with(converted_annotation, None, Path('custom'), None)

    def test_annotation_conversion_subset_size(self, mocker):
        addition_options = {
            'annotation_conversion': {'converter': 'wider', 'annotation_file': 'file'},
            'subsample_size': 1
        }
        config = copy_dataset_config(self.dataset_config)
        config.update(addition_options)
        converted_annotation = make_representation(['0 0 0 5 5', '0 1 1 10 10'], True)
        mocker.patch(
            'accuracy_checker.annotation_converters.WiderFormatConverter.convert',
            return_value=(converted_annotation, None)
        )
        dataset = Dataset(config, MockPreprocessor())
        assert dataset.annotation == [converted_annotation[1]]

    def test_annotation_conversion_subset_ratio(self, mocker):
        addition_options = {
            'annotation_conversion': {'converter': 'wider', 'annotation_file': 'file'},
            'subsample_size': '50%'
        }
        config = copy_dataset_config(self.dataset_config)
        config.update(addition_options)
        converted_annotation = make_representation(['0 0 0 5 5', '0 1 1 10 10'], True)
        mocker.patch(
            'accuracy_checker.annotation_converters.WiderFormatConverter.convert',
            return_value=(converted_annotation, None)
        )
        subset_maker_mock = mocker.patch(
            'accuracy_checker.dataset.make_subset'
        )
        Dataset(config, MockPreprocessor())
        subset_maker_mock.assert_called_once_with(converted_annotation, 1, 666)

    def test_annotation_conversion_subset_with_seed(self, mocker):
        addition_options = {
            'annotation_conversion': {'converter': 'wider', 'annotation_file': 'file'},
            'subsample_size': 1,
            'subsample_seed': 1
        }
        config = copy_dataset_config(self.dataset_config)
        config.update(addition_options)
        converted_annotation = make_representation(['0 0 0 5 5', '0 1 1 10 10'], True)
        mocker.patch(
            'accuracy_checker.annotation_converters.WiderFormatConverter.convert',
            return_value=(converted_annotation, None)
        )
        dataset = Dataset(config, MockPreprocessor())
        annotation = dataset.annotation
        assert annotation == [converted_annotation[0]]

    def test_annotation_conversion_save_subset(self, mocker):
        addition_options = {
            'annotation_conversion': {'converter': 'wider', 'annotation_file': 'file'},
            'annotation': 'custom',
            'subsample_size': 1,
        }
        config = copy_dataset_config(self.dataset_config)
        config.update(addition_options)
        converted_annotation = make_representation(['0 0 0 5 5', '0 1 1 10 10'], True)
        mocker.patch(
            'accuracy_checker.annotation_converters.WiderFormatConverter.convert',
            return_value=(converted_annotation, None)
        )
        annotation_saver_mock = mocker.patch(
            'accuracy_checker.dataset.save_annotation'
        )
        Dataset(config, MockPreprocessor())
        annotation_saver_mock.assert_called_once_with([converted_annotation[1]], None, Path('custom'), None)
