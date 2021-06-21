# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

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
            graph_def, _, _ = load_tf_graph_def('path', False)
            summary = summarize_graph(graph_def)
            self.assertEqual(len(summary['outputs']), 1)
            self.assertEqual(summary['outputs'][0], 'Output/Identity')
            self.assertEqual(len(summary['inputs']), 1)
            self.assertEqual('Placeholder' in summary['inputs'], True)
            self.assertEqual(str(summary['inputs']['Placeholder']['shape']), '(1,227,227,3)')
            self.assertEqual(str(summary['inputs']['Placeholder']['type']), 'float32')
