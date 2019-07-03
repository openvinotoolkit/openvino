"""
 Copyright (c) 2018-2019 Intel Corporation

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

from mo.front.tf.extractors.concat import tf_concat_ext
from mo.utils.unittest.extractors import PB, BaseExtractorsTestingClass


class ConcatExtractorTest(BaseExtractorsTestingClass):
    @classmethod
    def setUpClass(cls):
        cls.patcher = 'mo.front.tf.extractors.concat.concat_infer'

    def test_concat(self):
        pb = PB({'attr': {
            'N': PB({'i': 3}),
        }})
        self.expected = {
            'type': 'Concat',
            'N': 3,
        }
        self.res = tf_concat_ext(pb=pb)
        self.res["infer"](None)
        self.call_args = self.infer_mock.call_args
        self.expected_call_args = (None)
        self.compare()
