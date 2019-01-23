"""
 Copyright (c) 2018 Intel Corporation

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

from mo.front.tf.extractors.matmul import tf_matmul_ext
from mo.utils.unittest.extractors import PB, BaseExtractorsTestingClass


class MatmulExtractorTest(BaseExtractorsTestingClass):
    @classmethod
    def setUpClass(cls):
        cls.patcher = 'mo.front.tf.extractors.matmul.tf_matmul_infer'

    def test_matmul(self):
        pb = PB({'attr': {
            'transpose_a': PB({
                'b': True
            }),
            'transpose_b': PB({
                'b': False
            }),
        }})
        self.expected = {
            'transpose_a': True,
            'transpose_b': False,
        }
        self.res = tf_matmul_ext(pb=pb)
        self.res["infer"](None)
        self.call_args = self.infer_mock.call_args
        self.expected_call_args = None
        self.compare()
