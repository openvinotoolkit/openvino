# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import io
import os
import struct
from xml.etree import ElementTree

import numpy as np
import pytest

import openvino as ov
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


class CacheBoundary(Op):
    class_type_info = DiscreteTypeInfo("CacheBoundary", "cache_boundary_test")

    def __init__(self, inputs=None, mode=0):
        super().__init__(self)
        self._attrs = {"mode": mode}
        if inputs is not None:
            self.set_arguments(inputs)
            self.constructor_validate_and_infer_types()

    def get_type_info(self):
        return self.class_type_info

    def visit_attributes(self, visitor):
        visitor.on_attributes(self._attrs)
        return True

    def clone_with_new_inputs(self, inputs):
        return CacheBoundary(inputs, self._attrs["mode"])

    def validate_and_infer_types(self):
        if self._attrs["mode"] == 0:
            for port in (0, 1):
                self.set_output_type(port, self.get_input_element_type(port), self.get_input_partial_shape(port))
        else:
            source = 1 if self._attrs["mode"] == 1 else 0
            self.set_output_type(0, self.get_input_element_type(source), self.get_input_partial_shape(source))


@gpu_only
def test_custom_cache_versions_outer_descriptor(device):
    core = Core()
    input_parameter = ops.parameter([1, 4], ov.Type.f32)
    compiled = core.compile_model(Model([ops.relu(input_parameter)], [input_parameter]), device)
    requirements = compiled.get_property("RUNTIME_REQUIREMENTS")
    assert requirements.startswith("meta=1.1;")
    exported = compiled.export_model().getvalue()
    magic = struct.pack("=Q", 0x4F5645505F525251)
    version_offset = exported.index(magic) + len(magic)
    assert struct.unpack_from("=I", exported, version_offset)[0] == 3
    previous_version = bytearray(exported)
    struct.pack_into("=I", previous_version, version_offset, 2)
    with pytest.raises(RuntimeError, match="Unsupported compatibility descriptor version"):
        core.import_model(io.BytesIO(previous_version), device)


def check_custom_cache_round_trip(device, tmp_path, model, config, samples, extension=None):
    config = dict(config, INFERENCE_PRECISION_HINT=ov.Type.f32, NUM_STREAMS="1", CACHE_DIR=str(tmp_path / "cache"))
    core = Core()
    if extension is not None:
        core.add_extension(ov.OpExtension(extension))
    cold = core.compile_model(model, device, config)
    assert not cold.get_property("LOADED_FROM_CACHE")
    for inputs, expected in samples:
        actual = list(cold(inputs).values())
        assert len(actual) == len(expected)
        for output, reference in zip(actual, expected):
            assert output.shape == reference.shape
            assert output.dtype == reference.dtype
            np.testing.assert_allclose(output, reference, rtol=1e-6, atol=1e-6)
    blob = cold.export_model()
    import_config = dict(config)
    import_config.pop("CACHE_DIR")
    imported = core.import_model(blob, device, import_config)
    warm = core.compile_model(model, device, config)
    assert warm.get_property("LOADED_FROM_CACHE")
    for compiled in (imported, warm):
        for inputs, expected in samples:
            actual = list(compiled(inputs).values())
            assert len(actual) == len(expected)
            for output, reference in zip(actual, expected):
                assert output.shape == reference.shape
                assert output.dtype == reference.dtype
                np.testing.assert_allclose(output, reference, rtol=1e-6, atol=1e-6)


@gpu_only
@pytest.mark.parametrize("selected", [0, 1])
@pytest.mark.parametrize("rank", [2, 4])
def test_custom_cache_selected_input_work_sizes(device, tmp_path, selected, rank):
    selected_parameter = ops.parameter([1] * (rank - 1) + [4], ov.Type.f32)
    dynamic_parameter = ops.parameter([1, 1, 1, -1], ov.Type.f32)
    params = [selected_parameter, dynamic_parameter] if selected == 0 else [dynamic_parameter, selected_parameter]
    node = CacheBoundary(params, 1 if selected == 0 else 2)
    model = Model([node], params)
    kernel = """__kernel void dispatch(__global const float* a, __global const float* b,
        __global float* out) {
        // Observe the launch size without allowing the old oversized launch to read out of bounds.
        for (size_t i=get_global_id(0); i<OUTPUT0_DIMS[3]; i+=get_global_size(0))
            out[i]=(float)get_global_size(0);
    }"""
    config = write_config(tmp_path, f"""<CustomLayer name="CacheBoundary" type="SimpleGPU" version="1">
        <Kernel entry="dispatch"><Source filename="dispatch.cl"/></Kernel><Buffers>
        <Tensor arg-index="0" type="input" port-index="0" format="BFYX"/>
        <Tensor arg-index="1" type="input" port-index="1" format="BFYX"/>
        <Tensor arg-index="2" type="output" port-index="0" format="BFYX"/></Buffers>
        <WorkSizes dim="input,{selected}" global="X,1,1" local="1,1,1"/>
        </CustomLayer>""", {"dispatch.cl": kernel})
    samples = []
    for width in (8, 12):
        fixed = np.zeros([1] * (rank - 1) + [4], np.float32)
        dynamic_values = np.zeros([1, 1, 1, width], np.float32)
        inputs = [fixed, dynamic_values] if selected == 0 else [dynamic_values, fixed]
        samples.append((inputs, [np.full(dynamic_values.shape, 4, np.float32)]))
    check_custom_cache_round_trip(device, tmp_path, model, {"CONFIG_FILE": config}, samples, CacheBoundary)


@gpu_only
def test_custom_cache_mixed_static_dynamic_outputs(device, tmp_path):
    dynamic_parameter = ops.parameter([-1, 4], ov.Type.f32)
    static_parameter = ops.parameter([4], ov.Type.f32)
    node = CacheBoundary([dynamic_parameter, static_parameter])
    model = Model(node.outputs(), [dynamic_parameter, static_parameter])
    kernel = """__kernel void mixed(__global const float* a, __global const float* b,
        __global float* out_a, __global float* out_b) {
        size_t i=get_global_id(0); out_a[i]=a[i]; if(i<4)out_b[i]=b[i];
    }"""
    config = write_config(tmp_path, """<CustomLayer name="CacheBoundary" type="SimpleGPU" version="1">
        <Kernel entry="mixed"><Source filename="mixed.cl"/></Kernel><Buffers>
        <Tensor arg-index="0" type="input" port-index="0" format="BFYX"/>
        <Tensor arg-index="1" type="input" port-index="1" format="BFYX"/>
        <Tensor arg-index="2" type="output" port-index="0" format="BFYX"/>
        <Tensor arg-index="3" type="output" port-index="1" format="BFYX"/></Buffers>
        <WorkSizes global="B*F*Y*X,1,1" local="1,1,1"/></CustomLayer>""", {"mixed.cl": kernel})
    samples = []
    for batch in (1, 3, 2):
        first = np.arange(batch * 4, dtype=np.float32).reshape(batch, 4)
        second = np.arange(4, dtype=np.float32) + 20
        samples.append(([first, second], [first, second]))
    check_custom_cache_round_trip(device, tmp_path, model, {"CONFIG_FILE": config}, samples, CacheBoundary)


@gpu_only
def test_custom_cache_internal_atan2(device, tmp_path):
    first = ops.parameter([-1, 4], ov.Type.f32)
    second = ops.parameter([-1, 4], ov.Type.f32)
    zero = ops.constant(np.float32(0))
    angle = ops.atan(ops.divide(first, second))
    pi_value = ops.constant(np.float32(np.pi))
    negative_x = ops.select(ops.greater_equal(first, zero), ops.add(angle, pi_value), ops.subtract(angle, pi_value))
    nonzero_x = ops.select(ops.greater(second, zero), angle, negative_x)
    result = ops.select(ops.equal(second, zero), ops.multiply(ops.sign(first), np.float32(np.pi / 2)), nonzero_x)
    model = Model([result], [first, second])
    kernel = """__kernel void atan2_custom(__global const float* y, __global const float* x,
        __global float* out) {size_t i=get_global_id(0);out[i]=atan2(y[i],x[i]);}"""
    config = write_config(tmp_path, """<CustomLayer name="Atan2" type="SimpleGPU" version="1">
        <Kernel entry="atan2_custom"><Source filename="atan2.cl"/></Kernel><Buffers>
        <Tensor arg-index="0" type="input" port-index="0" format="BFYX"/>
        <Tensor arg-index="1" type="input" port-index="1" format="BFYX"/>
        <Tensor arg-index="2" type="output" port-index="0" format="BFYX"/></Buffers>
        <WorkSizes global="B*F*Y*X,1,1" local="1,1,1"/></CustomLayer>""", {"atan2.cl": kernel})
    samples = []
    for batch in (1, 3, 2):
        values = np.tile(np.array([[-2, 1, 3, -4]], np.float32), (batch, 1))
        divisors = np.tile(np.array([[1, -2, 4, -3]], np.float32), (batch, 1))
        samples.append(([values, divisors], [np.arctan2(values, divisors)]))
    check_custom_cache_round_trip(device, tmp_path, model, {"CONFIG_FILE": config}, samples)


@gpu_only
def test_custom_cache_lstm_peephole_input(device, tmp_path):
    from openvino.utils.node_factory import NodeFactory

    shapes = [[-1, 2], [-1, 2], [-1, 2], [8, 2], [8, 2], [8], [6]]
    params = [ops.parameter(shape, ov.Type.f32) for shape in shapes]
    node = NodeFactory("opset1").create("LSTMCell", params, {
        "hidden_size": 2, "activations": ["sigmoid", "tanh", "tanh"],
        "activations_alpha": [], "activations_beta": [], "clip": 0.0,
        "weights_format": "ifco", "input_forget": False,
    })
    model = Model(node.outputs(), params)
    kernel = """__kernel void lstm_custom(__global const float* x, __global const float* h,
        __global const float* c, __global const float* w, __global const float* r,
        __global const float* b, __global const float* p, __global float* out_h,
        __global float* out_c) {size_t i=get_global_id(0);out_h[i]=x[i]+p[0];out_c[i]=c[i]+p[1];}"""
    buffers = "".join(f'<Tensor arg-index="{index}" type="input" port-index="{index}" format="BFYX"/>'
                      for index in range(7))
    buffers += ('<Tensor arg-index="7" type="output" port-index="0" format="BFYX"/>'
                '<Tensor arg-index="8" type="output" port-index="1" format="BFYX"/>')
    config = write_config(tmp_path, f"""<CustomLayer name="LSTMCell" type="SimpleGPU" version="1">
        <Kernel entry="lstm_custom"><Source filename="lstm.cl"/></Kernel><Buffers>{buffers}</Buffers>
        <WorkSizes global="B*F*Y*X,1,1" local="1,1,1"/></CustomLayer>""", {"lstm.cl": kernel})
    samples = []
    for batch in (1, 3, 2):
        inputs = []
        for shape in shapes:
            data_shape = [batch if dim < 0 else dim for dim in shape]
            inputs.append(np.arange(np.prod(data_shape), dtype=np.float32).reshape(data_shape))
        inputs[6] += 5
        samples.append((inputs, [inputs[0] + inputs[6][0], inputs[2] + inputs[6][1]]))
    check_custom_cache_round_trip(device, tmp_path, model, {"CONFIG_FILE": config}, samples)


@gpu_only
def test_custom_cache_identical_shape_models_are_independent(device, tmp_path):
    first = ops.parameter([-1, 4], ov.Type.f32)
    second = ops.parameter([-1, 4], ov.Type.f32)
    model = Model([ops.relu(first), ops.relu(second)], [first, second])
    kernel = """__kernel void relu_custom(__global const float* data, __global float* out) {
        size_t i=get_global_id(0);out[i]=fmax(data[i],0.0f);
    }"""
    config = write_config(tmp_path, """<CustomLayer name="Relu" type="SimpleGPU" version="1">
        <Kernel entry="relu_custom"><Source filename="relu.cl"/></Kernel><Buffers>
        <Tensor arg-index="0" type="input" port-index="0" format="BFYX"/>
        <Tensor arg-index="1" type="output" port-index="0" format="BFYX"/></Buffers>
        <WorkSizes global="B*F*Y*X,1,1" local="1,1,1"/></CustomLayer>""", {"relu.cl": kernel})
    samples = []
    for first_batch, second_batch in ((1, 3), (3, 1), (2, 4)):
        first_values = np.arange(first_batch * 4, dtype=np.float32).reshape(first_batch, 4) - 3
        second_values = np.arange(second_batch * 4, dtype=np.float32).reshape(second_batch, 4) - 5
        samples.append(([first_values, second_values], [np.maximum(first_values, 0), np.maximum(second_values, 0)]))
    check_custom_cache_round_trip(device, tmp_path, model, {"CONFIG_FILE": config}, samples)


@gpu_only
def test_gpu_load_keeps_internal_atan2_private(device, tmp_path):
    first = ops.parameter([-1, 4], ov.Type.f32)
    second = ops.parameter([-1, 4], ov.Type.f32)
    xml_path = tmp_path / "model.xml"
    ov.serialize(Model([ops.add(first, second)], [first, second]), xml_path)
    document = ElementTree.parse(xml_path)
    add_layer = next(layer for layer in document.findall("./layers/layer") if layer.get("type") == "Add")
    add_layer.set("type", "Atan2")
    add_layer.set("version", "ie_internal_opset")
    document.write(xml_path)
    core = Core()
    core.get_property(device, "FULL_DEVICE_NAME")
    # Loading GPU must not expose its private operation factories to ordinary Core readers.
    with pytest.raises(RuntimeError, match="Atan2"):
        core.read_model(xml_path)
