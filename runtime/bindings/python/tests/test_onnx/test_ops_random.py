# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import onnx
import onnx.mapping
import pytest
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info

from ngraph.exceptions import NgraphTypeError
from tests.runtime import get_runtime
from tests.test_onnx.utils import get_node_model, import_onnx_model, run_model, run_node


def test_random_normal():
    node = onnx.helper.make_node("RandomNormal", inputs=[], outputs=["y"], mean=100.0, scale=10.0, seed=5.0, shape=(30, 30))
    result = run_node(node, [])[0]
    from IPython import embed; embed()

    assert result.shape == (30, 30)
    assert np.allclose(result, np.eye(shape[0], shape[1], k=shift))
