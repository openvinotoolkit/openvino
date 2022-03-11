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
import re
import sys
import logging as log
from common.samples_common_test_clas import get_tests
from common.samples_common_test_clas import SamplesCommonTestClass

log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

test_data = get_tests(cmd_params={'sample_type': ['Python', 'C++']})

class TestHelloQueryDevice(SamplesCommonTestClass):
    @classmethod
    def setup_class(cls):
        cls.sample_name = 'hello_query_device'
        super().setup_class()

    @pytest.mark.parametrize("param", test_data)
    def test_hello_query_device(self, param):
        """
        Hello Query Device test has functional and accuracy tests.
        For accuracy find in output line available devices
        """

        # Run _test function, that returns stdout or 0.
        stdout = self._test(param, use_preffix=False, get_cmd_func=self.get_empty_cmd_line)
        if not stdout:
            return 0
        stdout = stdout.split('\n')
        is_ok = 0
        for line in stdout:
            log.info(line)
            if "Available devices" in line:
                is_ok = True
        assert is_ok != 0, "Sample stdout didn't match"
