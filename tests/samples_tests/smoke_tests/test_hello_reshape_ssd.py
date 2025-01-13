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
from common.samples_common_test_class import get_tests
from common.samples_common_test_class import SamplesCommonTestClass
from common.specific_samples_parsers import parse_hello_reshape_ssd


test_data_fp32 = get_tests({
    'i': ['samples_smoke_tests_data_2021.4/validation_set/500x500/cat.bmp'],
    'm': ['samples_smoke_tests_data_2021.4/models/public/ssd512/FP16/ssd512.xml'],
    'sample_type': ['C++','Python'],
    'd': ['CPU']}, use_device=['d'], use_batch=False)


class TestHelloShape(SamplesCommonTestClass):
    sample_name = 'hello_reshape_ssd'

    @pytest.mark.parametrize("param", test_data_fp32)
    def test_hello_reshape_ssd_fp32(self, param, cache):
        assert parse_hello_reshape_ssd(self._test(param, cache, use_preffix=False, get_cmd_func=self.get_hello_shape_cmd_line).split('\n'))
