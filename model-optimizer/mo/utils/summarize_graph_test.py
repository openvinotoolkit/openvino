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

import unittest
from unittest.mock import patch, mock_open

from mo.front.tf.loader import load_tf_graph_def
from mo.utils.summarize_graph import summarize_graph

pbtxt = 'node{name:"Placeholder"op:"Placeholder"attr{key:"dtype"value{type:DT_FLOAT}}attr{key:"shape"value{shape{dim' + \
        '{size:1}dim{size:227}dim{size:227}dim{size:3}}}}}node{name:"Output/Identity"op:"Identity"input:"Placeholder' + \
        '"attr{key:"T"value{type:DT_FLOAT}}}'


class TestingSummarizeGraph(unittest.TestCase):
    def test_summarize_graph(self):
        with patch('mo.front.tf.loader.open', mock_open(read_data=pbtxt)) as m:
            graph_def, _ = load_tf_graph_def('path', False)
            summary = summarize_graph(graph_def)
            self.assertEqual(len(summary['outputs']), 1)
            self.assertEqual(summary['outputs'][0], 'Output/Identity')
            self.assertEqual(len(summary['inputs']), 1)
            self.assertEqual('Placeholder' in summary['inputs'], True)
            self.assertEqual(str(summary['inputs']['Placeholder']['shape']), '(1,227,227,3)')
            self.assertEqual(str(summary['inputs']['Placeholder']['type']), 'float32')
