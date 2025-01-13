"""
 Copyright (C) 2018-2025 Intel Corporation
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
import sys
import logging as log
from common.samples_common_test_class import SamplesCommonTestClass
from common.samples_common_test_class import get_tests

log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

test_data_fp32 = get_tests({
    'i': ['dog-224x224.bmp'],
    'm': ['bvlcalexnet-12.onnx'],  # Remove the model forom .md and .rst if removed from here
    'sample_type': ['C++','Python'],
})

test_data_fp16 = get_tests({
    'i': ['dog-224x224.bmp'],
    'm': ['bvlcalexnet-12.onnx'],
    'sample_type': ['C++','Python'],
})


class TestClassification(SamplesCommonTestClass):
    sample_name = 'classification_sample_async'

    @pytest.mark.parametrize("param", test_data_fp32)
    def test_classification_sample_async_fp32(self, param, cache):
        _check_output(self, param, '215', cache)

    @pytest.mark.parametrize("param", test_data_fp16)
    def test_classification_sample_async_fp16(self, param, cache):
        _check_output(self, param, '215', cache)


def _check_output(self, param, expected_result, cache):
    """
    Classification_sample_async has functional and accuracy tests.
    For accuracy find in output class of detected on image object
    """
    # Run _test function, that returns stdout or 0.
    stdout = self._test(param, cache)
    if not stdout:
        return 0

    stdout = stdout.split('\n')
    is_ok = 0
    for line in range(len(stdout)):
        if re.match(r"\d+ +\d+.\d+$", stdout[line].replace('[ INFO ]', '').strip()) is not None:
            if is_ok == 0:
                is_ok = True
            top1 = stdout[line].replace('[ INFO ]', '').strip().split(' ')[0]
            top1 = re.sub(r"\D", "", top1)
            if expected_result not in top1:
                is_ok = False
                log.info("Detected class {}".format(top1))
            break
    assert is_ok != 0, "Accuracy check didn't passed, probably format of output has changes"
    assert is_ok, "Wrong top1 class"
    log.info('Accuracy passed')
