# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import copy
import pytest
import numpy as np
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema_fb
from tensorflow.lite.tools import flatbuffer_utils as utils

from common.tflite_layer_test_class import TFLiteLayerTest

SCALE = 0.0039
ZERO_POINT = 0


class TestTFLiteQuantizeMultiOutputs(TFLiteLayerTest):
    """Test that TFLQuantizeConvert pass correctly handles quantize with multiple Convert consumers"""

    inputs = ["Input_0", "Input_1"]
    outputs = ["Result_0", "Result_1"]
    allowed_ops = ['ADD', 'QUANTIZE']

    def make_model(self, params):
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            input_0 = tf.compat.v1.placeholder(tf.float32, [1, 8], name=self.inputs[0])
            input_1 = tf.compat.v1.placeholder(tf.float32, [1, 8], name=self.inputs[1])

            add_out = tf.add(input_0, input_1, name="add_inputs")
            add_const_out = tf.add(add_out, tf.constant(10.0, shape=[1, 8], dtype=tf.float32), name="add_const_10")

            tf.identity(add_out, name=self.outputs[0])
            tf.identity(add_const_out, name=self.outputs[1])
            net = sess.graph_def
        return net

    def produce_tflite_model(self, framework_model, save_path):
        graph = tf.Graph()
        with graph.as_default():
            tf.import_graph_def(framework_model, name='')

        with tf.compat.v1.Session(graph=graph) as sess:
            input_tensors = [graph.get_tensor_by_name(f'{name}:0') for name in self.inputs]
            output_tensors = [graph.get_tensor_by_name(f'{name}:0') for name in self.outputs]

            converter = tf.compat.v1.lite.TFLiteConverter.from_session(sess, input_tensors, output_tensors)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8

            def representative_dataset():
                for _ in range(128):
                    input_0 = np.random.uniform(0.0, 0.995, [1, 8]).astype(np.float32)
                    input_1 = np.random.uniform(0.0, 0.995, [1, 8]).astype(np.float32)
                    yield [input_0, input_1]

            converter.representative_dataset = representative_dataset
            tflite_model = converter.convert()

        # Post-process to strip QUANTIZE wrapper ops and keep only ADD ops with quantized tensors
        model_obj = utils.read_model_from_bytearray(tflite_model)
        self._strip_quantize_wrappers(model_obj)
        self._force_u8_quant_params(model_obj)

        tflite_model_path = os.path.join(save_path, 'quantize_multi_outs.tflite')
        utils.write_model(model_obj, tflite_model_path)

        return tflite_model_path

    def _force_u8_quant_params(self, model_obj):
        """Force uint8 quantization parameters on all uint8 tensors."""
        subgraph = model_obj.subgraphs[0]
        for tensor in subgraph.tensors:
            if tensor.type != schema_fb.TensorType.UINT8:
                continue
            if tensor.quantization is None:
                tensor.quantization = schema_fb.QuantizationParametersT()
            tensor.quantization.scale = [float(SCALE)]
            tensor.quantization.zeroPoint = [int(ZERO_POINT)]
            tensor.quantization.min = []
            tensor.quantization.max = []

    def _strip_quantize_wrappers(self, model_obj):
        """Remove QUANTIZE ops, keeping only ADD operations with quantized tensors."""
        subgraph = model_obj.subgraphs[0]
        # Find all ADD operators
        add_indices = []
        for i, op in enumerate(subgraph.operators):
            code = model_obj.operatorCodes[op.opcodeIndex].builtinCode
            if code == schema_fb.BuiltinOperator.ADD:
                add_indices.append(i)

        if len(add_indices) < 2:
            raise RuntimeError("Expected at least two ADD operators")

        # Keep the two ADD operations
        add_inputs_op = copy.deepcopy(subgraph.operators[add_indices[0]])
        add_const_op = copy.deepcopy(subgraph.operators[add_indices[1]])

        # Wire them directly: first ADD uses model inputs, second ADD uses first's output
        add_inputs_op.inputs = [subgraph.inputs[0], subgraph.inputs[1]]
        add_const_op.inputs = [add_inputs_op.outputs[0], add_const_op.inputs[1]]

        subgraph.operators = [add_inputs_op, add_const_op]

        # Set model outputs to the ADD operation outputs
        add_out_idx = add_inputs_op.outputs[0]
        add_const_out_idx = add_const_op.outputs[0]
        const_idx = add_const_op.inputs[1]
        subgraph.outputs = [add_out_idx, add_const_out_idx]

        # Name and quantize all relevant tensors
        for idx, name in [
            (subgraph.inputs[0], b"Input_0"),
            (subgraph.inputs[1], b"Input_1"),
            (add_out_idx, b"Result_0"),
            (add_const_out_idx, b"Result_1"),
            (const_idx, b"const_10"),
        ]:
            tensor = subgraph.tensors[idx]
            tensor.name = name
            tensor.type = schema_fb.TensorType.UINT8

        # Set constant tensor as uint8 with value 10
        const_tensor = subgraph.tensors[const_idx]
        const_tensor.shape = [1, 8]
        const_buffer = model_obj.buffers[const_tensor.buffer]
        const_buffer.data = bytearray([10] * 8)

    def _prepare_input(self, inputs_dict, generator=None):
        """Generate uint8 input data matching model input type."""
        for input_name, input_shape in inputs_dict.items():
            inputs_dict[input_name] = np.random.randint(0, 256, size=input_shape, dtype=np.uint8)
        return inputs_dict

    @pytest.mark.nightly
    def test_quantize_multi_outputs(self, ie_device, precision, temp_dir):
        """
        Test that quantized model with multiple outputs produces correct results.
        This verifies that TFLQuantizeConvert pass correctly optimizes when quantize has
        multiple Convert consumers, preserving output types and runtime info.
        """
        params = {}
        self._test(ie_device, precision, temp_dir, params)
