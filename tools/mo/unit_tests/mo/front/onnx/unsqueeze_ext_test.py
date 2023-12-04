# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

import numpy as np
import onnx

from openvino.tools.mo.front.onnx.unsqueeze_ext import UnsqueezeFrontExtractor
from openvino.tools.mo.ops.op import Op
from openvino.tools.mo.ops.unsqueeze import Unsqueeze
from unit_tests.utils.extractors import PB


class TestUnsqueezeONNXExt():
    @staticmethod
    def _create_unsqueeze_node(axes):
        if axes is None:
            pb = onnx.helper.make_node(
                'Unsqueeze',
                inputs=['x'],
                outputs=['y'],
            )
        else:
            pb = onnx.helper.make_node(
                'Unsqueeze',
                inputs=['x'],
                outputs=['y'],
                axes=axes,
            )

        node = PB({'pb': pb})
        return node

    @classmethod
    def setUpClass(cls):
        Op.registered_ops['Unsqueeze'] = Unsqueeze

    @pytest.mark.parametrize("axes",[[0, 1, 2, 3], [1], []])
    def test_unsqueeze_ext(self, axes):
        node = self._create_unsqueeze_node(axes)
        UnsqueezeFrontExtractor.extract(node)

        exp_res = {
            'expand_axis': axes,
        }

        for key in exp_res.keys():
            if type(node[key]) in [list, np.ndarray]:
                assert np.array_equal(np.array(node[key]), np.array(exp_res[key]))
            else:
                assert node[key] == exp_res[key]
