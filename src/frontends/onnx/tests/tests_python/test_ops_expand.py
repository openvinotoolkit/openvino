# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import tempfile
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import openvino as ov


def test_expand_empty_shape_keeps_scalar():
    x = helper.make_tensor_value_info("x", TensorProto.INT32, [])
    y = helper.make_tensor_value_info("y", TensorProto.INT32, [])
    shape_init = numpy_helper.from_array(np.array([], dtype=np.int64), name="shape")
    node = helper.make_node("Expand", ["x", "shape"], ["y"])
    graph = helper.make_graph([node], "expand_empty_shape_scalar", [x], [y], [shape_init])
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=10,
    )

    onnx_path = Path(tempfile.gettempdir()) / "expand_empty_shape_scalar.onnx"
    onnx_path.write_bytes(model.SerializeToString())

    compiled = ov.compile_model(str(onnx_path), "CPU")
    out = compiled({"x": np.array(-8, dtype=np.int32)})[compiled.output(0)]

    assert tuple(out.shape) == ()
    assert np.array_equal(np.asarray(out), np.array(-8, dtype=np.int32))
