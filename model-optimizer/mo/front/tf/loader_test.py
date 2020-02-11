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

import unittest.mock
from io import StringIO
from unittest.mock import Mock, MagicMock

from generator import generate, generator

from mo.front.tf.loader import load_tf_graph_def


@generator
class TestLoader(unittest.TestCase):
    @generate('/path/to/somewhere/my_checkpoint.ckpt', '/path/to/somewhere/my_meta_graph.meta')
    @unittest.mock.patch('sys.stdout', new_callable=StringIO)
    def test_helper_print_ckpt(self, path, out):
        mock = Mock(__bool__=MagicMock(side_effect=Exception()))
        self.assertRaises(Exception, load_tf_graph_def, path, meta_graph_file=mock)
        self.assertRegex(out.getvalue(),
                         '\[ WARNING ] The value for the --input_model command line parameter ends with "\.ckpt"')
