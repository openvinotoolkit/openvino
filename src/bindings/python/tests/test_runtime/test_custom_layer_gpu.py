# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
import pytest

import openvino.opset14 as ops
from openvino import Core, DiscreteTypeInfo, Model, Op, Shape

gpu_only = pytest.mark.skipif(
    os.environ.get("TEST_DEVICE") not in ["GPU"],
    reason="Device dependent test",
)


def make_custom_op_type(type_name):
    """Build an Op whose type name the GPU plugin resolves against a CustomLayer entry."""

    class CustomLayerOp(Op):
        class_type_info = DiscreteTypeInfo(type_name, "gpu_opset")

        def __init__(self, inputs=None):
            super().__init__(self)
            if inputs is not None:
                self.set_arguments(inputs)
                self.constructor_validate_and_infer_types()

        def validate_and_infer_types(self):
            self.set_output_type(0, self.get_input_element_type(0), self.get_input_partial_shape(0))

        def clone_with_new_inputs(self, new_inputs):
            return CustomLayerOp(new_inputs)

        def get_type_info(self):
            return CustomLayerOp.class_type_info

    return CustomLayerOp


def write_config(tmp_path, layer_xml, kernels):
    for name, source in kernels.items():
        (tmp_path / name).write_text(source)
    config = tmp_path / "custom_layers.xml"
    config.write_text(layer_xml)
    return str(config)


AXIS_PROBE_KERNEL = """
__kernel void axis_probe(__global const INPUT0_TYPE* input, __global OUTPUT0_TYPE* output) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    output[y * OUTPUT0_DIMS[3] + x] = (OUTPUT0_TYPE)(y * 1000 + x);
}
"""

COPY_KERNEL = """
__kernel void copy(__global const INPUT0_TYPE* input, __global OUTPUT0_TYPE* output) {
    const int id = get_global_id(0);
    output[id] = input[INPUT0_OFFSET + id];
}
"""

ADD_KERNEL = """
__kernel void add(__global const INPUT0_TYPE* a, __global const INPUT1_TYPE* b,
                  __global OUTPUT0_TYPE* output) {
    const int id = get_global_id(0);
    output[id] = a[id] + b[id];
}
"""


@gpu_only
def test_custom_layer_output_format_any_keeps_axis_order(device, tmp_path):
    # format="ANY" on an output means "no conversion reorder after me". It must not change
    # the axis order the op's own WorkSizes resolution, or the rest of the graph, sees.
    # The shape is non-square and the dispatch is over X and Y separately, so a transposed
    # layout cannot cancel out.
    config = write_config(
        tmp_path,
        """<CustomLayer name="CustomLayerAnyFmt" type="SimpleGPU" version="1">
    <Kernel entry="axis_probe">
        <Source filename="axis_probe.cl"/>
    </Kernel>
    <Buffers>
        <Tensor arg-index="0" type="input" port-index="0"/>
        <Tensor arg-index="1" type="output" port-index="0" format="ANY"/>
    </Buffers>
    <WorkSizes global="X,Y,1"/>
</CustomLayer>""",
        {"axis_probe.cl": AXIS_PROBE_KERNEL},
    )

    shape = [1, 1, 3, 4]
    param = ops.parameter(Shape(shape), dtype=np.float32, name="data")
    custom = make_custom_op_type("CustomLayerAnyFmt")([param.output(0)])
    model = Model([custom.output(0)], [param], "any_fmt")

    compiled = Core().compile_model(model, device, {"CONFIG_FILE": config})
    result = compiled(np.zeros(shape, dtype=np.float32))[0]

    expected = np.array([[y * 1000 + x for x in range(4)] for y in range(3)], dtype=np.float32)
    assert np.array_equal(result.reshape(3, 4), expected)


@gpu_only
def test_custom_layer_reads_offset_slice(device, tmp_path):
    # The custom layer consumes the second half of a Split, i.e. a non-zero-offset view of
    # its producer. The view is optimized in place, so the kernel is handed the parent
    # buffer and reaches the cropped data through INPUT0_OFFSET, as the CustomLayer
    # documentation prescribes.
    config = write_config(
        tmp_path,
        """<CustomLayer name="CustomLayerCopy" type="SimpleGPU" version="1">
    <Kernel entry="copy">
        <Source filename="copy.cl"/>
    </Kernel>
    <Buffers>
        <Tensor arg-index="0" type="input" port-index="0"/>
        <Tensor arg-index="1" type="output" port-index="0"/>
    </Buffers>
    <WorkSizes global="B*F*Y*X,1,1"/>
</CustomLayer>""",
        {"copy.cl": COPY_KERNEL},
    )

    shape = [1, 4, 1, 2]
    param = ops.parameter(Shape(shape), dtype=np.float32, name="data")
    axis = ops.constant(np.int32(1))
    upper_half = ops.split(param, axis, 2).output(1)
    custom = make_custom_op_type("CustomLayerCopy")([upper_half])
    model = Model([custom.output(0)], [param], "offset_slice")

    data = np.array([0, 1, 10, 11, 20, 21, 30, 31], dtype=np.float32).reshape(shape)
    compiled = Core().compile_model(model, device, {"CONFIG_FILE": config})
    result = compiled(data)[0]

    assert np.array_equal(result.reshape(-1), np.array([20, 21, 30, 31], dtype=np.float32))


@gpu_only
def test_custom_layer_same_producer_on_two_ports(device, tmp_path):
    # Both input ports declare a format, so a pre-reorder is inserted for each. When one
    # producer feeds both, the two reorders must still get distinct primitive ids.
    config = write_config(
        tmp_path,
        """<CustomLayer name="CustomLayerAdd" type="SimpleGPU" version="1">
    <Kernel entry="add">
        <Source filename="add.cl"/>
    </Kernel>
    <Buffers>
        <Tensor arg-index="0" type="input" port-index="0" format="BFYX"/>
        <Tensor arg-index="1" type="input" port-index="1" format="BFYX"/>
        <Tensor arg-index="2" type="output" port-index="0" format="BFYX"/>
    </Buffers>
    <WorkSizes global="B*F*Y*X,1,1"/>
</CustomLayer>""",
        {"add.cl": ADD_KERNEL},
    )

    shape = [1, 2, 3, 4]
    param = ops.parameter(Shape(shape), dtype=np.float32, name="data")
    custom = make_custom_op_type("CustomLayerAdd")([param.output(0), param.output(0)])
    model = Model([custom.output(0)], [param], "shared_producer")

    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    compiled = Core().compile_model(model, device, {"CONFIG_FILE": config})
    result = compiled(data)[0]

    assert np.array_equal(result, data * 2)
