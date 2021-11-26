"""
 Copyright (C) 2018-2021 Intel Corporation
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
import pytest
import sys
import logging as log
from common.samples_common_test_clas import SamplesCommonTestClass
from common.samples_common_test_clas import get_tests
from common.common_comparations import check_image_if_box
from common.samples_common_test_clas import Environment

log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

test_data_ssd_person_and_bicycles_detection_fp32 = \
    get_tests(cmd_params={'i': [os.path.join('any', 'person_detection.png')],
                          'm': [os.path.join('FP32', 'pedestrian-detection-adas-0002.xml'),
                                os.path.join('FP32', 'person-vehicle-bike-detection-crossroad-1016.xml'),
                                os.path.join('FP32', 'pedestrian-and-vehicle-detector-adas-0001.xml')],
                          'batch': [1],
                          'd': ['CPU'],
                          'sample_type': ['C++', 'C']},
              use_device=['d']
              )

test_data_ssd_person_and_bicycles_detection_vehicle_fp32 = \
    get_tests(cmd_params={'i': [os.path.join('any', 'car.bmp')],
                          'm': [os.path.join('FP32', 'vehicle-detection-adas-0002.xml')],
                          'batch': [1],
                          'd': ['CPU'],
                          'sample_type': ['C++', 'C']},
              use_device=['d']
              )

class TestObjectDetectionSSD(SamplesCommonTestClass):
    @classmethod
    def setup_class(cls):
        cls.sample_name = 'object_detection_sample_ssd'
        super().setup_class()

    # The test above exesutes 3 different models:
    # person-vehicle-bike-detection-crossroad-1016,
    # pedestrian-detection-adas-0002,
    # pedestrian-and-vehicle-detector-adas-0001,
    # with the same parameters
    #
    # This test check
    #     1) sample draw something on output image
    @pytest.mark.parametrize("param", test_data_ssd_person_and_bicycles_detection_fp32)
    def test_object_detection_sample_ssd_person_and_bicycles_detection_fp32(self, param):
        _check_simple_output(self, param)

    @pytest.mark.parametrize("param", test_data_ssd_person_and_bicycles_detection_vehicle_fp32)
    def test_object_detection_sample_ssd_person_and_bicycles_detection_vehicle_fp32(self, param):
        _check_simple_output(self, param)


def _check_simple_output(self, param, empty_outputs=False):
    """
    Object_detection_sample_ssd has functional and accuracy testing.
    For accuracy comparing several metrics (reference file collected in some IE):
    -check that demo draw something in output image
    """

    # Run _test function, that returns stdout or 0.
    stdout = self._test(param)
    if not stdout:
        return
    stdout = stdout.split('\n')
    # This test check if boxes exist on output image (just that it draw something)
    img_path1 = ''
    img_path2 = param['i']
    for line in stdout:
        if "created!" in line:
            img_path1 = line.split(' ')[-2]
    acc_pass = check_image_if_box(os.path.join(os.getcwd(), img_path1),
                                  os.path.join(Environment.env['test_data'], img_path2))

    if not empty_outputs:
        assert acc_pass != 0, "Sample didn't draw boxes"
    else:
        assert acc_pass == 0, "Sample did draw boxes"

    log.info('Accuracy passed')


def _check_dog_class_output(self, param):
    """
    Object_detection_sample_ssd has functional and accuracy testing.
    For accuracy comparing several metrics (reference file collected in some IE):
    -check that demo draw something in output image
    -label of detected object with 100% equality
    """

    # Run _test function, that returns stdout or 0.
    stdout = self._test(param)
    if not stdout:
        return 0
    stdout = stdout.split('\n')
    # This test check if boxes exist on output image (just that it draw something)
    img_path1 = ''
    img_path2 = param['i']

    for line in stdout:
        if "created!" in line:
            img_path1 = line.split(' ')[-2]
    acc_pass = check_image_if_box(os.path.join(os.getcwd(), img_path1),
                                  os.path.join(Environment.env['test_data'], img_path2))
    assert acc_pass != 0, "Sample didn't draw boxes"

    # Check top1 class
    dog_class = '58'
    is_ok = 0
    for line in stdout:
        if 'WILL BE PRINTED' in line:
            is_ok += 1
            top1 = line.split(' ')[0]
            assert dog_class in top1, "Wrong top1 class, current {}, reference {}".format(top1, dog_class)
            log.info('Accuracy passed')
            break
    assert is_ok != 0, "Accuracy check didn't passed, probably format of output has changes"
