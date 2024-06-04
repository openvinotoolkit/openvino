# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest.mock
from io import StringIO
from openvino.tools.mo.front.tf.loader import load_tf_graph_def
from unittest.mock import Mock, MagicMock


class TestLoader(unittest.TestCase):
    @unittest.mock.patch('sys.stdout', new_callable=StringIO)
    def test_helper_print_ckpt(self, out):
        for path in ['/path/to/somewhere/my_checkpoint.ckpt', '/path/to/somewhere/my_meta_graph.meta']:
            mock = Mock(__bool__=MagicMock(side_effect=Exception()))
            self.assertRaises(Exception, load_tf_graph_def, path, meta_graph_file=mock)
            self.assertRegex(out.getvalue(),
                             r'\[ WARNING ] The value for the --input_model command line parameter ends with "\.ckpt"')
