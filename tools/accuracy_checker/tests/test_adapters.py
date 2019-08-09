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

import numpy as np
import pytest

from accuracy_checker.adapters import SSDAdapter, Adapter
from accuracy_checker.config import ConfigError
from .common import make_representation


def test_detection_adapter():
    raw = {
        'detection_out': np.array([[[[0, 3, 0.2, 0, 0, 1, 1], [0, 2, 0.5, 4, 4, 7, 7], [0, 5, 0.7, 3, 3, 9, 8]]]])
    }
    expected = make_representation('0.2,3,0,0,1,1;0.5,2,4,4,7,7;0.7,5,3,3,9,8')

    actual = SSDAdapter({}, output_blob='detection_out').process([raw], ['0'], [{}])

    assert np.array_equal(actual, expected)


def test_detection_adapter_partially_filling_output_blob():
    raw = {
        'detection_out': np.array(
            [[[[0, 3, 0.2, 0, 0, 1, 1], [0, 2, 0.5, 4, 4, 7, 7], [0, 5, 0.7, 3, 3, 9, 8], [-1, 0, 0, 0, 0, 0, 0]]]]
        )
    }
    expected = make_representation('0.2,3,0,0,1,1;0.5,2,4,4,7,7;0.7,5,3,3,9,8')

    actual = SSDAdapter({}, output_blob='detection_out').process([raw], ['0'])

    assert np.array_equal(actual, expected)


def test_detection_adapter_partially_filling_output_blob_with_zeros_at_the_end():
    raw = {
        'detection_out': np.array([[[
            [0,  3, 0.2, 0, 0, 1, 1],
            [0,  2, 0.5, 4, 4, 7, 7],
            [0,  5, 0.7, 3, 3, 9, 8],
            [-1, 0, 0,   0, 0, 0, 0],
            [0,  0, 0,   0, 0, 0, 0]
        ]]])
    }
    expected = make_representation('0.2,3,0,0,1,1;0.5,2,4,4,7,7;0.7,5,3,3,9,8')

    actual = SSDAdapter({}, output_blob='detection_out').process([raw], ['0'])

    assert np.array_equal(actual, expected)


def test_detection_adapter_batch_2():
    raw = {
        'detection_out': np.array([[[[0, 3, 0.2, 0, 0, 1, 1], [0, 2, 0.5, 4, 4, 7, 7], [1, 5, 0.7, 3, 3, 9, 8]]]])
    }
    expected = make_representation(['0.2,3,0,0,1,1;0.5,2,4,4,7,7', '0.7,5,3,3,9,8'])

    actual = SSDAdapter({}, output_blob='detection_out').process([raw], ['0', '1'])

    assert np.array_equal(actual, expected)


def test_dictionary_adapter_no_raise_warning_on_specific_args():
    adapter_config = {'type': 'age_gender', 'gender_out': 'gender', 'age_out': 'age'}
    with pytest.warns(None) as record:
        Adapter.provide('age_gender', adapter_config)
        assert len(record) == 0


def test_age_gender_adapter_raise_config_error_on_extra_args():
    adapter_config = {'type': 'age_gender', 'gender_out': 'gender', 'age_out': 'age', 'something_extra': 'extra'}
    with pytest.raises(ConfigError):
        Adapter.provide('age_gender', adapter_config)


def test_face_person_detection_adapter_raise_config_error_on_extra_args():
    adapter_config = {
        'type': 'face_person_detection',
        'face_detection_out': 'face',
        'person_detection_out': 'person',
        'something_extra': 'extra'
    }
    with pytest.raises(ConfigError):
        Adapter.provide('face_person_detection', adapter_config)


def test_head_pose_adapter_raise_config_error_on_extra_args():
    adapter_config = {
        'type': 'head_pose',
        'angle_yaw': 'yaw',
        'angle_pitch': 'pitch',
        'angle_roll': 'roll',
        'something_extra': 'extra'
    }
    with pytest.raises(ConfigError):
        Adapter.provide('head_pose', adapter_config)


def test_vehicle_attributes_adapter_raise_config_error_on_extra_args():
    adapter_config = {
        'type': 'vehicle_attributes',
        'color_out': 'color',
        'type_out': 'type',
        'something_extra': 'extra'
    }
    with pytest.raises(ConfigError):
        Adapter.provide('vehicle_attributes', adapter_config)
