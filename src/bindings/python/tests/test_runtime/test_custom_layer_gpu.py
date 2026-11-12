# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import io
import os
import struct

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


class CacheConcat(ov.Op):
    """Public test extension with a serialized shape-relevant attribute."""

    class_type_info = ov.DiscreteTypeInfo("CacheConcat", "cache_test_extension")

    def __init__(self, inputs=None, axis=1):
        super().__init__(self)
        self._attrs = {"axis": int(axis)}
        if inputs is not None:
            self.set_arguments(inputs)
            self.constructor_validate_and_infer_types()

    def get_type_info(self):
        return self.class_type_info

    def visit_attributes(self, visitor):
        visitor.on_attributes(self._attrs)
        return True

    def clone_with_new_inputs(self, inputs):
        return CacheConcat(inputs, self._attrs["axis"])

    def validate_and_infer_types(self):
        inferred = ops.concat(self.input_values(), self._attrs["axis"])
        self.set_output_type(0, inferred.get_output_element_type(0), inferred.get_output_partial_shape(0))


@gpu_only
@pytest.mark.parametrize("mode", ["path", "precision", "axis", "nested", "loop", "loop_invariant", "extension"])
def test_custom_layer_dynamic_cache_restoration(device, tmp_path, mode):
    if mode == "precision" and "FP16" not in Core().get_property(device, "OPTIMIZATION_CAPABILITIES"):
        pytest.skip("The precision cache case requires FP16 support")
    assets = tmp_path / "assets"
    assets.mkdir()
    concat = mode in ("axis", "extension")
    half = mode == "precision"
    name = "CacheConcat" if mode == "extension" else ("Concat" if concat else "Relu")
    scalar = "half" if half else "float"
    code = "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n" if half else ""
    if concat:
        code += (
            "__kernel void custom(__global const float* x,__global const float* y,"
            "__global float* z){size_t i=get_global_id(0),n=get_global_size(0)/2;"
            "z[i]=i<n?x[i]:y[i-n];}\n"
        )
        buffers = (
            '<Tensor arg-index="0" type="input" port-index="0" format="BFYX"/><'
            'Tensor arg-index="1" type="input" port-index="1" format="BFYX"/><'
            'Tensor arg-index="2" type="output" port-index="0" format="BFYX"/>'
        )
    else:
        code += (
            f"__kernel void custom(__global const {scalar}* x,"
            f"__global {scalar}* y){{size_t i=get_global_id(0);y[i]=fmax(x[i],({scalar})0);}}\n"
        )
        buffers = (
            '<Tensor arg-index="0" type="input" port-index="0" format="BFYX"/><'
            'Tensor arg-index="1" type="output" port-index="0" format="BFYX"/>'
        )
    (assets / "custom.cl").write_text(code)
    xml = (
        f'<CustomLayer name="{name}" type="SimpleGPU" version="1"><Kernel entry="custom"><'
        f'Source filename="custom.cl"/></Kernel><Buffers>{buffers}</Buffers><'
        f'WorkSizes global="B*F*Y*X,1,1" local="1,1,1"/></CustomLayer>'
    )
    (assets / "custom.xml").write_text(xml)
    input_parameter = ops.parameter(ov.PartialShape([-1, -1] if concat else [-1, 4]), ov.Type.f32, name="input")
    params = [input_parameter]
    if concat:
        second = ops.parameter(ov.PartialShape([-1, -1]), ov.Type.f32, name="second")
        params.append(second)
        result = CacheConcat(params, 0) if mode == "extension" else ops.concat(params, 0)
        result.set_friendly_name("custom_op")
    elif mode == "nested":
        condition = ops.parameter([], ov.Type.boolean, name="condition")
        params.append(condition)
        body_p = ops.parameter([-1, 4], ov.Type.f32, name="body_input")
        yes = ov.Model([ops.relu(body_p, name="custom_op")], [body_p])
        no_p = ops.parameter([-1, 4], ov.Type.f32, name="else_input")
        else_body = ov.Model([ops.negative(no_p)], [no_p])
        branch = ops.if_op(condition)
        branch.set_then_body(yes)
        branch.set_else_body(else_body)
        branch.set_input(input_parameter.output(0), body_p, no_p)
        branch.set_output(yes.get_results()[0], else_body.get_results()[0])
        branch.validate_and_infer_types()
        result = branch
    elif mode in ("loop", "loop_invariant"):
        trip = ops.parameter([], ov.Type.i64, name="trip_count")
        params.append(trip)
        body_p = ops.parameter([-1, 4], ov.Type.f32, name="body_input")
        activation = ops.relu(body_p, name="custom_op")
        condition = ops.constant(np.array(True))
        body = ov.Model([condition, activation], [body_p])
        loop = ops.loop(trip, ops.constant(np.array(True)))
        loop.set_function(body)
        if mode == "loop_invariant":
            loop.set_invariant_input(body_p, input_parameter.output(0))
        else:
            loop.set_merged_input(body_p, input_parameter.output(0), activation.output(0))
        loop.set_special_body_ports([-1, 0])
        result = loop.get_iter_value(activation.output(0), -1)
        loop.validate_and_infer_types()
    else:
        source = ops.add(input_parameter, ops.constant(np.float32(1))) if half else input_parameter
        result = ops.relu(source, name="custom_op")
        if half:
            result = ops.multiply(result, ops.constant(np.float32(2)))
    model = ov.Model([result], params, "CustomLayerCacheRestoration")
    config = {
        "CONFIG_FILE": str(assets / "custom.xml"),
        "INFERENCE_PRECISION_HINT": ov.Type.f16 if half else ov.Type.f32,
        "NUM_STREAMS": "1",
        "CACHE_DIR": str(tmp_path / "cache"),
    }
    ov.serialize(model, assets / "model.xml", assets / "model.bin")

    def compile_one():
        core = Core()
        if mode == "extension":
            core.add_extension(ov.OpExtension(CacheConcat))
        compiled = core.compile_model(
            str(assets / "model.xml") if mode == "path" else model,
            device,
            config,
        )
        return core, compiled

    core, cold = compile_one()
    assert not cold.get_property("LOADED_FROM_CACHE")
    exported = cold.export_model()
    blob = exported.getvalue() if hasattr(exported, "getvalue") else bytes(exported)
    del cold, core
    core, warm = compile_one()
    assert warm.get_property("LOADED_FROM_CACHE"), "warm load silently recompiled"
    import_config = dict(config)
    import_config.pop("CACHE_DIR")
    if concat:
        # A conflicting hint must not replace the serialized axis=0 operation.
        wrong = CacheConcat(params, 1) if mode == "extension" else ops.concat(params, 1)
        wrong.set_friendly_name("custom_op")
        import_config["MODEL_PTR"] = Model([wrong], params)
    imported = core.import_model(io.BytesIO(blob), device, import_config)
    for batch in (1, 3, 2):
        width = 3 if concat else 4
        values = np.arange(batch * width, dtype=np.float32).reshape(batch, width) - 4
        inputs = [values]
        if concat:
            inputs.append(values + 20)
            expected = np.concatenate(inputs, axis=0)
        else:
            expected = np.maximum(values + (1 if half else 0), 0) * (2 if half else 1)
        if mode == "nested":
            inputs.append(np.array(True))
        elif mode in ("loop", "loop_invariant"):
            inputs.append(np.array(3, dtype=np.int64))
        for compiled in (warm, imported):
            outputs = compiled(inputs)
            actual = np.array(outputs[compiled.output(0)], copy=True)
            assert actual.shape == expected.shape
            assert actual.dtype == expected.dtype
            assert actual.tobytes() == expected.tobytes()
        if mode == "nested":
            for compiled in (warm, imported):
                outputs = compiled([values, np.array(False)])
                actual = np.array(outputs[compiled.output(0)], copy=True)
                assert actual.shape == values.shape
                assert actual.tobytes() == (-values).tobytes()


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
