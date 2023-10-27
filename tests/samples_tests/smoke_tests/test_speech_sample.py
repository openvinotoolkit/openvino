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
import sys
import logging as log
from common.samples_common_test_class import SamplesCommonTestClass
from common.samples_common_test_class import Environment
from common.samples_common_test_class import get_tests
from common.common_utils import parse_avg_err

log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

test_data = get_tests(cmd_params={'i': [os.path.join('ark', 'dev93_10.ark')],
                                           'm': [os.path.join('wsj', 'FP32', 'wsj_dnn5b.xml')],
                                           'layout': ["[NC]"],
                                           'bs': [1, 2],
                                           'o': ['res_output.ark'],
                                           'r': [os.path.join('ark', 'dev93_scores_10.ark')],
                                           'qb': [8, 16],
                                           'sf': ["2175.43"],
                                           'q': ["static", "user"],
                                           'd': ['GNA_SW_EXACT']},
                               use_device=False
                               )
							   
new_format_test_data = get_tests(cmd_params={'i': ['Parameter=' + os.path.join(Environment.env['test_data'], 'ark', 'dev93_10.ark')],
                                           'm': [os.path.join('wsj', 'FP32', 'wsj_dnn5b.xml')],
                                           'layout': ["[NC]"],
                                           'bs': [1],
                                           'o': ['affinetransform14/Fused_Add_:0=' + os.path.join(Environment.env['test_data'], 'res_output.ark')],
                                           'r': ['affinetransform14/Fused_Add_:0=' + os.path.join(Environment.env['test_data'], 'ark', 'dev93_scores_10.ark')],
                                           'qb': [8],
                                           'sf': ["Parameter=2175.43"],
                                           'q': ["static"],
                                           'memory_reuse_off': [" "],
                                           'd': ['GNA_SW_EXACT']},
                               use_device=False
                               )

class TestSpeechSample(SamplesCommonTestClass):
    @classmethod
    def setup_class(cls):
        cls.sample_name = 'speech_sample'
        cls.threshold = 0.06
        super().setup_class()

    @pytest.mark.parametrize("param", test_data)
    @pytest.mark.skipif(sys.platform == 'darwin', reason="GNA is not available on macOS")
    def test_speech_sample_nthreads(self, param):
        stdout = self._test(param).split('\n')

        avg_error = parse_avg_err(stdout)
        log.info('Average scores diff: {}'.format(avg_error))
        assert avg_error <= self.threshold
		
    @pytest.mark.parametrize("param", new_format_test_data)
    @pytest.mark.skipif(sys.platform == 'darwin', reason="GNA is not available on macOS")
    def test_speech_sample_new_format(self, param):
        stdout = self._test(param, complete_path=False).split('\n')

        avg_error = parse_avg_err(stdout)
        log.info('Average scores diff: {}'.format(avg_error))
        assert avg_error <= self.threshold
