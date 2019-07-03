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

import unittest

import numpy as np
from extensions.front.tf.stop_gradient_ext import StopGradientExtractor
from mo.utils.unittest.extractors import PB
from generator import generator, generate


from mo.utils.unittest.extractors import PB, BaseExtractorsTestingClass


class StopGradientTest(BaseExtractorsTestingClass):

    def test_stop_gradient(self):
        node = PB({
            'pb': PB({
                'attr': PB({
                    'T': PB({
                        "type": 1
                    })
                })
            })
        })
        self.expected = {
            'op': 'StopGradient'
        }
        StopGradientExtractor().extract(node)
        self.res = node
        self.compare()

