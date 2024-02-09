"""
 Copyright (C) 2018-2023 Intel Corporation
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
from common.samples_common_test_class import SamplesCommonTestClass
from common.samples_common_test_class import get_tests

test_data_fp32_async = get_tests \
    (cmd_params={'i': [os.path.join('227x227', 'dog.bmp')],
                 'm': [os.path.join('squeezenet1.1', 'FP32', 'squeezenet1.1.xml')],
                 'batch': [1],
                 'sample_type': ['C++', 'Python'],
                 'api': ['async'],
                 'nireq': ['4'],
                 'niter': ['10'], },
     )

test_data_fp32_sync = get_tests \
    (cmd_params={'i': [os.path.join('227x227', 'dog.bmp')],
                 'm': [os.path.join('squeezenet1.1', 'FP32', 'squeezenet1.1.xml')],
                 'batch': [1],
                 'sample_type': ['C++', 'Python'],
                 'niter': ['10'],
                 'api': ['sync']},
     )


class Test_benchmark_app(SamplesCommonTestClass):
    @classmethod
    def setup_class(cls):
        cls.sample_name = 'benchmark_app'
        super().setup_class()

    @pytest.mark.parametrize("param", test_data_fp32_async + test_data_fp32_sync)
    def test(self, param):
        stdout = self._test(param)
        assert "FPS" in stdout
        assert "Latency" in stdout
