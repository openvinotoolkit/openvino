# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest.mock
from io import StringIO
from unittest.mock import Mock, MagicMock


from openvino.tools.mo.front.tf.loader import load_tf_graph_def


class TestLoader(unittest.TestCase):
    def test_helper_print_ckpt(self):
        test_cases = [
            ('/path/to/somewhere/my_checkpoint.ckpt', '/path/to/somewhere/my_meta_graph.meta'),
            # Add more test cases as needed
        ]

        for idx, (path, meta_graph_file) in enumerate(test_cases):
            with self.subTest(test_case=f"Test case {idx+1}"):
                with unittest.mock.patch('sys.stdout', new_callable=StringIO) as out:
                    mock = Mock(__bool__=MagicMock(side_effect=Exception()))
                    self.assertRaises(Exception, load_tf_graph_def, path, meta_graph_file=mock)
                    self.assertRegex(out.getvalue(), r'\[ WARNING ] The value for the --input_model command line parameter ends with "\.ckpt"')
