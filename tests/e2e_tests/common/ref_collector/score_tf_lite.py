# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from e2e_tests.common.ref_collector.provider import ClassProvider


class ScoreTensorFLowLite(ClassProvider):
    __action_name__ = "score_tf_lite"

    def __init__(self, config):
        self.model = config["model"]
        self.inputs = config["inputs"]
        self.res = {}

    def get_refs(self):
        import tensorflow as tf
        interpreter = tf.compat.v1.lite.Interpreter(model_path=self.model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_name_to_id_mapping = {input['name']: input['index'] for input in input_details}

        for layer, data in self.inputs.items():
            tensor_index = input_name_to_id_mapping[layer]
            tensor_id = next(i for i, tensor in enumerate(input_details) if tensor['index'] == tensor_index)
            interpreter.set_tensor(input_details[tensor_id]['index'], data)

        interpreter.invoke()

        for output in output_details:
            self.res[output['name']] = interpreter.get_tensor(output['index'])

        return self.res
