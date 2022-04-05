"""
 Copyright (C) 2018-2022 Intel Corporation
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
import logging as log
import sys
from common.samples_common_test_clas import get_tests
from common.samples_common_test_clas import SamplesCommonTestClass
from common.specific_samples_parsers import parse_hello_reshape_ssd

log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

test_data_fp32 = get_tests(cmd_params={'i': [os.path.join('500x500', 'cat.bmp')],
                                       'm': [os.path.join('ssd512', 'FP32', 'ssd512.xml')],
                                       'sample_type': ['C++','Python'],
                                       'd': ['CPU']},
                                       use_device=['d'], use_batch=False
                           )


class TestHelloShape(SamplesCommonTestClass):
    @classmethod
    def setup_class(cls):
        cls.sample_name = 'hello_reshape_ssd'
        super().setup_class()

    @pytest.mark.parametrize("param", test_data_fp32)
    def test_hello_reshape_ssd_fp32(self, param):
        """
        Hello_reshape_ssd has functional testing.
        This function get stdout from hello_reshape_ssd (already splitted by new line)
        The test check not if resulted class of object is accurate with reference, but that demo detected class with its box
        and so on and so forth.
        """
        # Run _test function, that returns stdout or 0.
        stdout = self._test(param, use_preffix=False, get_cmd_func=self.get_hello_shape_cmd_line)
        if not stdout:
            return 0
        stdout = stdout.split('\n')

        is_ok = parse_hello_reshape_ssd(stdout)
        assert is_ok, "[ERROR] Check failed"
        log.info('Functional test passed')
