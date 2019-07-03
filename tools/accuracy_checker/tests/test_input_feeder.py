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

import pytest
import re
import numpy as np
from accuracy_checker.config import ConfigError
from accuracy_checker.launcher.input_feeder import InputFeeder
from accuracy_checker.dataset import DataRepresentation

# InputInfo from openvino is needed here, but there is no appropriate API
# to create InputInfo with specific shape, therefore lets use analog
class InputInfo_test:
    layout = ''
    precision = ''
    shape = []
    def __init__(self, layout = '', precision = '', shape = []):
        self.layout = layout
        self.precision = precision
        self.shape = shape

class TestInputFeeder:
    def test_create_input_feeder_without_inputs_raise_config_error(self):
        with pytest.raises(ConfigError):
            InputFeeder([], {})

    def test_create_input_feeder_with_config_inputs_and_empty_network_inputs_raise_config_error(self):
        with pytest.raises(ConfigError):
            InputFeeder([{'name': 'const_data', 'type': 'CONST_INPUT', 'value': '[1, 1, 1, 1]'}], {})

    def test_create_input_feeder_with_config_const_inputs_not_in_network_inputs_raise_config_error(self):
        with pytest.raises(ConfigError):
            InputFeeder([{'name': 'const_data', 'type': 'CONST_INPUT', 'value': '[1, 1, 1, 1]'}], {'data': (1, 3, 10, 10)})

    def test_create_input_feeder_with_config_inputs_not_in_network_inputs_raise_config_error(self):
        with pytest.raises(ConfigError):
            InputFeeder([{'name': 'data2', 'type': 'INPUT', 'value': '.'}], {'data': (1, 3, 10, 10)})

    def test_create_input_feeder_without_config_inputs(self):
        input_feeder = InputFeeder([], {'data': (1, 3, 10, 10)})
        assert not input_feeder.const_inputs
        assert not input_feeder.inputs_mapping
        assert input_feeder.non_constant_inputs == ['data']

    def test_create_input_feeder_config_inputs_fully_match_to_network_inputs(self):
        input_feeder = InputFeeder([{'name': 'data', 'type': 'INPUT', 'value': '.'}], {'data': (1, 3, 10, 10)})
        assert not input_feeder.const_inputs
        assert input_feeder.inputs_mapping == {'data': re.compile('.')}
        assert input_feeder.non_constant_inputs == ['data']

    def test_create_input_feeder_config_inputs_contain_only_const_inputs_with_list_value(self):
        input_feeder = InputFeeder([{'name': 'const_data', 'type': 'CONST_INPUT', 'value': [1, 1, 1, 1]}], {'data': (1, 3, 10, 10), 'const_data': (1, 4)})
        assert np.array_equal(input_feeder.const_inputs['const_data'], np.ones(4))
        assert not input_feeder.inputs_mapping
        assert input_feeder.non_constant_inputs == ['data']

    def test_create_input_feeder_config_inputs_contain_only_const_inputs_with_not_list_value(self):
        input_feeder = InputFeeder(
            [{'name': 'const_data', 'type': 'CONST_INPUT', 'value': 'value'}],
            {'data': (1, 3, 10, 10), 'const_data': (1, 4)}
        )
        assert input_feeder.const_inputs['const_data'] == 'value'
        assert not input_feeder.inputs_mapping
        assert input_feeder.non_constant_inputs == ['data']

    def test_create_input_feeder_not_all_non_constant_inputs_in_config_raise_config_error(self):
        with pytest.raises(ConfigError):
            InputFeeder(
                [{'name': '0', 'type': 'INPUT', 'value': '.'}],
                {'0': (1, 3, 10, 10), '1': (1, 3, 10, 10)}
            )

    def test_fill_non_constant_input_with_one_input_without_specific_mapping_batch_1(self):
        input_feeder = InputFeeder([], { 'input': InputInfo_test(shape=(1, 3, 10, 10)) })
        result = input_feeder.fill_non_constant_inputs([DataRepresentation(np.zeros((10, 10, 3)), identifier='0')])[0]
        expected_data = np.zeros((1, 3, 10, 10))
        assert 'input' in result
        assert np.array_equal(result['input'], expected_data)

    def test_fill_non_constant_input_without_specific_mapping_batch_2(self):
        input_feeder = InputFeeder([], { 'input': InputInfo_test(shape=(1, 3, 10, 10))})
        result = input_feeder.fill_non_constant_inputs([
            DataRepresentation(np.zeros((10, 10, 3)), identifier='0'),
            DataRepresentation(np.zeros((10, 10, 3)), identifier='1')
        ])[0]
        expected_data = np.zeros((2, 3, 10, 10))
        assert 'input' in result
        assert np.array_equal(result['input'], expected_data)

    def test_fill_non_constant_input_with_specific_mapping_batch_1(self):
        input_feeder = InputFeeder([{'name': 'input', 'type': 'INPUT', 'value': '.'}], {'input': InputInfo_test(shape=(1, 3, 10, 10))})
        result = input_feeder.fill_non_constant_inputs([DataRepresentation(np.zeros((10, 10, 3)), identifier='0')])[0]
        expected_data = np.zeros((1, 3, 10, 10))
        assert 'input' in result
        assert np.array_equal(result['input'], expected_data)

    def test_fill_non_constant_input_with_specific_mapping_sevaral_image_matched(self):
        input_feeder = InputFeeder([{'name': 'input', 'type': 'INPUT', 'value': '.'}], {'input': InputInfo_test(shape=(1, 3, 10, 10))})
        result = input_feeder.fill_non_constant_inputs([DataRepresentation([np.zeros((10, 10, 3)), np.ones((10, 10, 3))], identifier=['0', '1'])])[0]
        expected_data = np.zeros((1, 3, 10, 10))
        assert 'input' in result
        assert np.array_equal(result['input'], expected_data)

    def test_fill_non_constant_input_with_specific_mapping_not_match_raise_config_error(self):
        input_feeder = InputFeeder([{'name': 'input', 'type': 'INPUT', 'value': '1.'}], {'input': InputInfo_test(shape=(1, 3, 10, 10))})
        with pytest.raises(ConfigError):
            input_feeder.fill_non_constant_inputs([DataRepresentation(np.zeros((10, 10, 3)), identifier='0')])

    def test_fill_non_constant_input_with_specific_mapping_batch_2(self):
        input_feeder = InputFeeder([{'name': 'input', 'type': 'INPUT', 'value': '.'}], {'input': InputInfo_test(shape=(1, 3, 10, 10))})
        result = input_feeder.fill_non_constant_inputs([
            DataRepresentation(np.zeros((10, 10, 3)), identifier='0'),
            DataRepresentation(np.zeros((10, 10, 3)), identifier='1')
        ])[0]
        expected_data = np.zeros((2, 3, 10, 10))
        assert 'input' in result
        assert np.array_equal(result['input'], expected_data)

    def test_fill_non_constant_input_with_specific_mapping_not_all_image_in_batch_matched_raise_config_error(self):
        input_feeder = InputFeeder([{'name': 'input', 'type': 'INPUT', 'value': '0+'}], {'input': InputInfo_test(shape=(1, 3, 10, 10))})
        with pytest.raises(ConfigError):
            input_feeder.fill_non_constant_inputs([
                DataRepresentation(np.zeros((10, 10, 3)), identifier='0'),
                DataRepresentation(np.zeros((10, 10, 3)), identifier='1')
            ])

    def test_fill_non_constant_inputs_without_specific_mapping_batch_1(self):
        input_feeder = InputFeeder([], { 'input1': InputInfo_test(shape=(1, 3, 10, 10)), 'input2': InputInfo_test(shape=(1, 3, 10, 10))})
        result = input_feeder.fill_non_constant_inputs([DataRepresentation(np.zeros((10, 10, 3)), identifier='0')])[0]
        expected_data = np.zeros((1, 3, 10, 10))
        assert 'input1' in result
        assert np.array_equal(result['input1'], expected_data)
        assert 'input2' in result
        assert np.array_equal(result['input2'], expected_data)

    def test_fill_non_constant_inputs_without_specific_mapping_batch_2(self):
        input_feeder = InputFeeder([], {'input1': InputInfo_test(shape=(1, 3, 10, 10)), 'input2': InputInfo_test(shape = (1, 3, 10, 10))})
        result = input_feeder.fill_non_constant_inputs([
            DataRepresentation(np.zeros((10, 10, 3)), identifier='0'),
            DataRepresentation(np.zeros((10, 10, 3)), identifier='1')
        ])[0]
        expected_data = np.zeros((2, 3, 10, 10))
        assert 'input1' in result
        assert np.array_equal(result['input1'], expected_data)
        assert 'input2' in result
        assert np.array_equal(result['input2'], expected_data)

    def test_fill_non_constant_inputs_with_specific_mapping_batch_1(self):
        input_feeder = InputFeeder(
            [{'name': 'input1', 'type': 'INPUT', 'value': '0'}, {'name': 'input2', 'type': 'INPUT', 'value': '1'}],
            {'input1': InputInfo_test(shape=(1, 3, 10, 10)), 'input2': InputInfo_test(shape=(1, 3, 10, 10))}
        )
        result = input_feeder.fill_non_constant_inputs(
            [DataRepresentation([np.zeros((10, 10, 3)), np.ones((10, 10, 3))],identifier=['0', '1'])]
        )[0]
        expected_data = [np.zeros((1, 3, 10, 10)), np.ones((1, 3, 10, 10))]
        assert 'input1' in result
        assert np.array_equal(result['input1'], expected_data[0])
        assert 'input2' in result
        assert np.array_equal(result['input2'], expected_data[1])

    def test_fill_non_constant_inputs_with_specific_mapping_not_match_raise_config_error(self):
        input_feeder = InputFeeder(
            [{'name': 'input1', 'type': 'INPUT', 'value': '0'}, {'name': 'input2', 'type': 'INPUT', 'value': '1'}],
            {'input1': InputInfo_test(shape=(1, 3, 10, 10)), 'input2': InputInfo_test(shape=(1, 3, 10, 10))}
        )
        with pytest.raises(ConfigError):
            input_feeder.fill_non_constant_inputs([DataRepresentation([np.zeros((10, 10, 3)), np.ones((10, 10, 3))], identifier=['0', '2'])])

    def test_fill_non_constant_inputs_with_specific_mapping_batch_2(self):
        input_feeder = InputFeeder(
            [{'name': 'input1', 'type': 'INPUT', 'value': '0'}, {'name': 'input2', 'type': 'INPUT', 'value': '1'}],
            { 'input1': InputInfo_test(shape = (1, 3, 10, 10)), 'input2': InputInfo_test(shape=(1, 3, 10, 10))}
        )
        result = input_feeder.fill_non_constant_inputs([
            DataRepresentation([np.zeros((10, 10, 3)), np.ones((10, 10, 3))], identifier=['0', '1']),
            DataRepresentation([np.zeros((10, 10, 3)), np.ones((10, 10, 3))], identifier=['0', '1'])
        ])[0]
        expected_data = [np.zeros((2, 3, 10, 10)), np.ones((2, 3, 10, 10))]
        assert 'input1' in result
        assert np.array_equal(result['input1'], expected_data[0])
        assert 'input2' in result
        assert np.array_equal(result['input2'], expected_data[1])

    def test_fill_non_constant_inputs_with_specific_mapping_not_all_image_in_batch_matched_raise_config_error(self):
        input_feeder = InputFeeder(
            [{'name': 'input1', 'type': 'INPUT', 'value': '0'}, {'name': 'input2', 'type': 'INPUT', 'value': '1'}],
            {'input1': (1, 3, 10, 10), 'input2': (1, 3, 10, 10)}
        )
        with pytest.raises(ConfigError):
            input_feeder.fill_non_constant_inputs([
                DataRepresentation([np.zeros((10, 10, 3)), np.ones((10, 10, 3))], identifier=['0', '1']),
                DataRepresentation([np.zeros((10, 10, 3)), np.ones((10, 10, 3))], identifier=['0', '2'])
            ])

    def test_fill_non_const_input_with_multi_infer_data_batch_1(self):
        input_feeder = InputFeeder({}, {'input': (1, 3, 10, 10)})
        result = input_feeder.fill_non_constant_inputs([
            DataRepresentation([np.zeros((10, 10, 3)), np.ones((10, 10, 3))], {'multi_infer': True}, identifier='0')
        ])
        expected = [{'input': np.zeros((1, 3, 10, 10))}, {'input': np.ones((1, 3, 10, 10))}]
        assert len(result) == len(expected)
        assert np.array_equal(result[0]['input'], expected[0]['input'])
        assert np.array_equal(result[1]['input'], expected[1]['input'])

    def test_fill_non_const_input_with_multi_infer_data_batch_2(self):
        input_feeder = InputFeeder({}, {'input': (2, 3, 10, 10)})
        result = input_feeder.fill_non_constant_inputs([
            DataRepresentation(
                [np.zeros((10, 10, 3)), np.ones((10, 10, 3))],
                {'multi_infer': True},
                identifier='0'
            ),
            DataRepresentation(
                [np.zeros((10, 10, 3)), np.ones((10, 10, 3))],
                {'multi_infer': True},
                identifier='1'
            ),
        ])
        expected = [{'input': np.zeros((2, 3, 10, 10))}, {'input': np.ones((2, 3, 10, 10))}]
        assert len(result) == len(expected)
        assert np.array_equal(result[0]['input'], expected[0]['input'])
        assert np.array_equal(result[1]['input'], expected[1]['input'])

    def test_fill_non_const_input_with_multi_infer_not_consistent_data_batch_2(self):
        input_feeder = InputFeeder({}, {'input': (2, 3, 10, 10)})
        result = input_feeder.fill_non_constant_inputs([
            DataRepresentation(
                [np.zeros((10, 10, 3))],
                {'multi_infer': True},
                identifier='0'
            ),
            DataRepresentation(
                [np.zeros((10, 10, 3)), np.ones((10, 10, 3))],
                {'multi_infer': True},
                identifier='1'
            ),
        ])
        expected = [{'input': np.zeros((2, 3, 10, 10))}, {'input': np.ones((1, 3, 10, 10))}]
        assert len(result) == len(expected)
        assert np.array_equal(result[0]['input'], expected[0]['input'])
        assert np.array_equal(result[1]['input'], expected[1]['input'])
