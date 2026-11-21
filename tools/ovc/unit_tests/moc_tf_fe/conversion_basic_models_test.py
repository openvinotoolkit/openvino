# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import os
import pytest
from openvino import Core
from openvino.tools.ovc.convert import convert_model


class TestTFOutputShapeMetadata:
    @pytest.mark.parametrize("add_shapes", [False, True])
    @pytest.mark.parametrize("from_file", [False, True])
    def test_relu(self, add_shapes, from_file, tmp_path):
        import tensorflow as tf

        with tf.Graph().as_default() as graph:
            value = tf.compat.v1.placeholder(tf.float32, [2, 3], name="value")
            tf.nn.relu(value, name="Relu")
            graph_def = graph.as_graph_def(add_shapes=add_shapes)
        relu_node = next(node for node in graph_def.node if node.name == "Relu")
        assert ("_output_shapes" in relu_node.attr) == add_shapes
        assert "shape" not in relu_node.attr
        assert "dtype" not in relu_node.attr

        input_model = graph_def
        if from_file:
            input_model = tf.io.write_graph(graph_def, str(tmp_path), "model.pb", as_text=False)
        model = convert_model(input_model)
        assert len(model.outputs) == 1
        assert model.output(0).get_names() == {"Relu:0"}
        assert list(model.output(0).get_shape()) == [2, 3]
        assert model.get_results()[0].input_value(0).get_node().get_type_name() == "Relu"

    @pytest.mark.parametrize("add_shapes", [False, True])
    @pytest.mark.parametrize("from_file", [False, True])
    @pytest.mark.parametrize("selected_output", [None, "Split:1"])
    def test_split(self, add_shapes, from_file, selected_output, tmp_path):
        import tensorflow as tf

        with tf.Graph().as_default() as graph:
            value = tf.compat.v1.placeholder(tf.float32, [8, 3], name="value")
            tf.raw_ops.Split(axis=0, value=value, num_split=4, name="Split")
            graph_def = graph.as_graph_def(add_shapes=add_shapes)
        input_model = graph_def
        if from_file:
            input_model = tf.io.write_graph(graph_def, str(tmp_path), "model.pb", as_text=False)
        model = convert_model(input_model, output=selected_output)
        expected_indices = range(4) if selected_output is None else [1]
        assert len(model.outputs) == len(expected_indices)
        for output, index in zip(model.outputs, expected_indices):
            assert output.get_names() == {f"Split:{index}"}
            assert list(output.get_shape()) == [2, 3]

    @pytest.mark.parametrize("add_shapes", [False, True])
    @pytest.mark.parametrize("shape", [[2, 3], [None, 3], None])
    @pytest.mark.parametrize("op_type", ["Relu", "Split"])
    def test_decoder(self, add_shapes, shape, op_type):
        import tensorflow as tf
        from openvino import PartialShape
        from openvino.frontend.tensorflow.node_decoder import TFGraphNodeDecoder

        with tf.Graph().as_default() as graph:
            value = tf.compat.v1.placeholder(tf.float32, shape, name="value")
            if op_type == "Relu":
                outputs = [tf.nn.relu(value, name="Relu")]
            else:
                outputs = tf.raw_ops.Split(axis=0, value=value, num_split=2, name="Split")
            graph_def = graph.as_graph_def(add_shapes=add_shapes)
        with tf.Graph().as_default() as imported_graph:
            tf.import_graph_def(graph_def, name="")
            decoder = TFGraphNodeDecoder(imported_graph.get_operation_by_name(op_type), True, False)
        # TensorFlow strips _output_shapes during import; retain the serialized metadata.
        decoder.m_node_def.CopyFrom(next(node for node in graph_def.node if node.name == op_type))
        original_attributes = set(decoder.m_node_def.attr)
        output_shapes = decoder.get_attribute("_output_shapes")
        if add_shapes:
            expected_shapes = [
                PartialShape.dynamic() if output.shape.rank is None else
                PartialShape([dim if dim is not None else -1 for dim in output.shape.as_list()])
                for output in outputs
            ]
            assert output_shapes == expected_shapes
        else:
            assert output_shapes.value is None
        assert set(decoder.m_node_def.attr) == original_attributes


class TestMoFreezePlaceholderTFFE():
    def basic(self, input_model, argv_input, inputs, dtype, expected, only_conversion=False):
        path = os.path.dirname(__file__)
        input_model = os.path.join(path, "test_models", input_model)

        try:
            model = convert_model(input_model, input=argv_input)
        except Exception as ex:
            self.fail("Model conversion failed due to error: {}".format(ex))

        if only_conversion:
            return

        ie = Core()
        exec_net = ie.compile_model(model, "CPU")
        req = exec_net.create_infer_request()
        results = req.infer(inputs)
        values = list(results.values())[0]
        if dtype is not None:
            assert values.dtype == dtype
        assert np.allclose(values, expected)

    @pytest.mark.parametrize("input_freezing_value, inputs, expected, dtype, only_conversion", [
        # TODO: Return this test when new 'cut' helper is introduced
        # (
        #        "add:0[3],z",
        #        {"add:0": np.array([4, 5, 6], dtype=np.float32), "z": np.array([1, 2, 3], dtype=np.float32)},
        #        np.array([4, 10, 18], dtype=np.float32),
        #        np.float32,
        #        None
        # ),
        # (
        #        "add:0{i32}[3],z{i32}",
        #        {"add:0": np.array([4, 5, 6], dtype=np.int32), "z": np.array([1, 2, 3], dtype=np.int32)},
        #        np.array([4, 10, 18], dtype=np.int32),
        #        np.int32,
        #        None
        # ),
    ])
    def test_cutting_fp32(self, input_freezing_value, inputs, expected, dtype, only_conversion):
        self.basic("model_three_inputs.pbtxt", input_freezing_value, inputs, dtype, expected,
                   only_conversion)

    @pytest.mark.parametrize("inputs, inputs_data, expected, dtype, only_conversion", [
        # TODO: Return this test when new 'cut' helper is introduced
        # (
        #         "x[1,4],y[4]",
        #         {"x": np.array([[3, 2, 1, 5]], dtype=np.int32), "y": np.array([0, -1, -7, 8], dtype=np.int32)},
        #         np.array([[3, 1, -6, 13]], dtype=np.int32),
        #         np.int32,
        #         None
        # ),
        # (
        #         "x,y",
        #         {"x": np.array([[-3, 20, 1]], dtype=np.int32), "y": np.array([[10, -11, -17]], dtype=np.int32)},
        #         np.array([[7, 9, -16]], dtype=np.int32),
        #         np.int32,
        #         None
        # ),
        (
                "x:0",
                {"x:0": np.array([[-3, 20, 1]], dtype=np.int32)},
                np.array([[-2, 22, 4], [1, 25, 7]], dtype=np.int32),
                np.int32,
                None
        ),
    ])
    def test_placeholder_with_default(self, inputs, inputs_data, expected, dtype, only_conversion):
        self.basic("placeholder_with_default.pbtxt", inputs, inputs_data, dtype, expected,
                   only_conversion)

    def test_conversion_model_oneshot_iterator_default(self):
        self.basic("model_oneshot_iterator.pbtxt", None, None, None, None, True)

    @pytest.mark.parametrize("input_freezing_value, inputs, expected, dtype", [
        (
                "in2{f32}->[0.0 0.0 0.0 0.0]",
                {"in1": np.array([[1.0, 2.0], [3.0, 4.0]])},
                np.array([[1.0, 2.0], [3.0, 4.0]]),
                np.float32,
        ),
        (
                "in2->[1.0 15.0 15.5 1.0]",
                {"in1": np.array([[2.0, 4.0], [12.0, 8.0]])},
                np.array([[3.0, 19.0], [27.5, 9.0]]),
                np.float32,
        ),
    ])
    @pytest.mark.skip(reason="109220: Use generating script for this test model instead of Git LFS")
    def test_conversion_model_with_non_standard_extension(self, input_freezing_value, inputs, expected, dtype):
        self.basic("model_fp32.frozen", input_freezing_value, inputs, dtype, expected, only_conversion=False)

    @pytest.mark.skip(reason="109220: Make TF FE to return the error")
    def test_conversion_dir_model(self):
        with self.assertRaisesRegex(Exception,
                                    "Internal error or inconsistent input model: the frontend supports "
                                    "only frozen binary protobuf format."):
            self.basic(".", None, None, None, None,
                       only_conversion=True)

    @pytest.mark.parametrize("inputs, expected, dtype", [
        (
                {"x:0": np.array([1, 2], dtype=np.int32), "y:0": np.array([4], dtype=np.int32)},
                np.array([-3, -2], dtype=np.int32),
                np.int32,
        ),
        (
                {"x:0": np.array([20, 25], dtype=np.int32), "y:0": np.array([10], dtype=np.int32)},
                np.array([30, 35], dtype=np.int32),
                np.int32,
        )
    ])
    def test_conversion_pbtxt_model_with_inference(self, inputs, expected, dtype):
        self.basic("model_with_if.pbtxt", None, inputs, dtype, expected, only_conversion=False)

    @pytest.mark.parametrize("model_name, argv_input, inputs, expected, dtype", [
        # new frontend
        (
                "model_add_with_undefined_constant.pbtxt",
                ("x:0", [2, 3]),
                {"x:0": np.array([[12, 13, 10], [11, 14, 16]], dtype=np.float32)},
                np.array([[12, 13, 10], [11, 14, 16]], dtype=np.float32),
                np.float32
        ),
        (
                "model_mul_with_undefined_constant.pbtxt",
                ("x:0", [2]),
                {"x:0": np.array([11, -12], dtype=np.int32)},
                np.array([0, 0], dtype=np.int32),
                np.int32
        ),
    ])
    def test_conversion_model_with_undefined_constant(self, model_name, argv_input, inputs, expected, dtype):
        self.basic(model_name, argv_input, inputs, dtype, expected, only_conversion=False)
