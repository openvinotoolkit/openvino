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

from mo.front.common.partial_infer.elemental import copy_shape_infer
from mo.front.tf.extractors.lrn import tf_lrn_ext
from mo.utils.unittest.extractors import PB


class LRNExtractorTest(unittest.TestCase):
    """
    Unit Test:
        1. test_bias_check - check if bias is not 1
        2. test_simple_check - check IE parameters calculations
        
    """

    def test_simple_check(self):
        # Input parameters for LRN extactor
        # taken from ICV AlexNet LRN layer
        pb = PB({'attr': {
            'alpha': PB({'f': 0.000019999999494757503}),
            'beta': PB({'f': 0.75}),
            'bias': PB({'f':2.0}),
            'depth_radius': PB({'i': 2}),
        }})
        res = tf_lrn_ext(pb)
        # Reference results for given parameters
        ref = {
            'type': 'LRN',
            'alpha': 9.999999747378752e-05,
            'beta': 0.75,
            'bias': 2.0,
            'local_size': 5,
            'infer': copy_shape_infer,
        }
        for attr in ref:
            self.assertEqual(res[attr], ref[attr])
