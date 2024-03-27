# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import onnx
import pytest

from openvino.tools.mo.front.onnx.squeeze_ext import SqueezeFrontExtractor
from openvino.tools.mo.ops.op import Op
from openvino.tools.mo.ops.squeeze import Squeeze
from unit_tests.utils.extractors import PB


class TestSqueezeONNXExt():
    @staticmethod
    def _create_squeeze_node(axes):
        if axes is None:
            pb = onnx.helper.make_node(
                'Squeeze',
                inputs=['x'],
                outputs=['y'],
            )
        else:
            pb = onnx.helper.make_node(
                'Squeeze',
                inputs=['x'],
                outputs=['y'],
                axes=axes,
            )

        node = PB({'pb': pb})
        return node

    @classmethod
    def setUpClass(cls):
        Op.registered_ops['Squeeze'] = Squeeze

    @pytest.mark.parametrize("axes",[[0, 1, 2, 3], [1], None])
    def test_squeeze_ext(self, axes):
        node = self._create_squeeze_node(axes)
        SqueezeFrontExtractor.extract(node)

        exp_res = {
            'type': 'Squeeze',
            'squeeze_dims': axes,
        }

        for key in exp_res.keys():
            if type(node[key]) in [list, np.ndarray]:
                assert np.array_equal(np.array(node[key]), np.array(exp_res[key]))
            else:
                assert node[key] ==  exp_res[key]
