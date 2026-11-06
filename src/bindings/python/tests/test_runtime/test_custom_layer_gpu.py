# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import io
import os

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
    p = ops.parameter(ov.PartialShape([-1, -1] if concat else [-1, 4]), ov.Type.f32, name="input")
    params = [p]
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
        no = ov.Model([ops.negative(no_p)], [no_p])
        branch = ops.if_op(condition)
        branch.set_then_body(yes)
        branch.set_else_body(no)
        branch.set_input(p.output(0), body_p, no_p)
        branch.set_output(yes.get_results()[0], no.get_results()[0])
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
            loop.set_invariant_input(body_p, p.output(0))
        else:
            loop.set_merged_input(body_p, p.output(0), activation.output(0))
        loop.set_special_body_ports([-1, 0])
        result = loop.get_iter_value(activation.output(0), -1)
        loop.validate_and_infer_types()
    else:
        source = ops.add(p, ops.constant(np.float32(1))) if half else p
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
