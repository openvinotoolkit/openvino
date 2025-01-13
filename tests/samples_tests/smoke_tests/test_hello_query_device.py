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
import sys
import logging as log
from common.samples_common_test_class import get_tests
from common.samples_common_test_class import SamplesCommonTestClass

log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

test_data = get_tests({'sample_type': ['Python', 'C++']}, use_device=False)

class TestHelloQueryDevice(SamplesCommonTestClass):
    sample_name = 'hello_query_device'

    @pytest.mark.parametrize("param", test_data)
    def test_hello_query_device(self, param, cache):
        """
        Hello Query Device test has functional and accuracy tests.
        For accuracy find in output line available devices
        """

        # Run _test function, that returns stdout or 0.
        stdout = self._test(param, cache, use_preffix=False)
        if not stdout:
            return 0
        stdout = stdout.split('\n')
        is_ok = 0
        for line in stdout:
            log.info(line)
            if "Available devices" in line:
                is_ok = True
        assert is_ok != 0, "Sample stdout didn't match"
