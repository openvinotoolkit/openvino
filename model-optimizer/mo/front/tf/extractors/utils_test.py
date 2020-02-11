"""
 Copyright (C) 2018-2020 Intel Corporation

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

from mo.front.tf.extractors.utils import collect_tf_attrs
from mo.utils.unittest.extractors import PB


class AttrParsingTest(unittest.TestCase):
    def test_simple_check(self):
        pb = PB({'attr': {
            'str': PB({'s': "aaaa"}),
            'int': PB({'i': 7}),
            'float': PB({'f': 2.0}),
            'bool': PB({'b': True}),
            'lisint': PB({'list': PB({'i': 5, 'i': 6})})}
        })

        res = collect_tf_attrs(pb.attr)

        # Reference results for given parameters
        ref = {
            'str': pb.attr['str'].s,
            'int': pb.attr['int'].i,
            'float': pb.attr['float'].f,
            'bool': pb.attr['bool'].b,
            'lisint': pb.attr['lisint'].list.i
        }
        for attr in ref:
            self.assertEqual(res[attr], ref[attr])
