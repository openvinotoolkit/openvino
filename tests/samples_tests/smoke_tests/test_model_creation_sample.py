"""
 Copyright (C) 2024 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the 'License');
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an 'AS IS' BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import collections
import os
import pytest
from common.samples_common_test_class import get_tests, SamplesCommonTestClass


class Test_model_creation_sample(SamplesCommonTestClass):
    sample_name = 'model_creation_sample'

    @pytest.mark.parametrize('param', get_tests({'sample_type': ['C++', 'Python']}))
    def test(self, param, cache):
        stdout = self._test(collections.OrderedDict(
            m=f'"{os.environ["WORKSPACE"]}/samples/cpp/model_creation_sample/lenet.bin"',
            **param,
        ), cache, use_preffix=False).split(sep='\n')
        target_line_index = -1
        for i, line in enumerate(stdout):
            if 'classid' in line:
                target_line_index = i + 2
        assert target_line_index != -1, 'Output format was changed! Check output format.'
        target_line = stdout[target_line_index].replace('[ INFO ]', '').split()
        target_classid = '9'
        assert target_line[0] == target_classid, 'Wrong Top1 class! Expected {target_classid} instead of {target_line[0]}'
        target_pred = 1
        assert float(target_line[1]) == target_pred, f'Wrong prediction! Expected {target_pred} instead of {target_line[1]}'
