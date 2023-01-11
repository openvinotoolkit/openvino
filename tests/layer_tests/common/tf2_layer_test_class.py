# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

from common.layer_test_class import CommonLayerTest

from tests.layer_tests.common.tf_layer_test_class import transpose_nhwc_to_nchw
from tests.layer_tests.common.utils.tf_utils import summarize_graph


def save_to_tf2_savedmodel(tf2_model, path_to_saved_tf2_model):
    import tensorflow as tf
    assert int(tf.__version__.split('.')[0]) >= 2, "TensorFlow 2 must be used for this suite validation"
    tf.keras.models.save_model(tf2_model, path_to_saved_tf2_model, save_format='tf')
    assert os.path.isdir(path_to_saved_tf2_model), "the model haven't been saved " \
                                                   "here: {}".format(path_to_saved_tf2_model)
    return path_to_saved_tf2_model


def save_tf2_saved_model_to_tflite(savedmodel):
    import tensorflow as tf

    converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model(savedmodel)
    tflite_model = converter.convert()

    tflite_model_path = os.path.join(os.path.dirname(savedmodel), 'model.tflite')
    with tf.io.gfile.GFile(tflite_model_path, 'wb') as f:
        f.write(tflite_model)

    return tflite_model_path


class CommonTF2LayerTest(CommonLayerTest):
    input_model_key = "saved_model_dir"

    def produce_model_path(self, framework_model, save_path):
        if not getattr(self, 'tflite', False):
            return save_to_tf2_savedmodel(framework_model, save_path)
        else:
            self.input_model_key = 'input_model'
            tf2_saved_model = save_to_tf2_savedmodel(framework_model, save_path)
            return save_tf2_saved_model_to_tflite(tf2_saved_model)

    def get_framework_results(self, inputs_dict, model_path):
        if not getattr(self, 'tflite', False):
            return self.get_tf2_keras_results(inputs_dict, model_path)
        else:
            # get results from tflite
            return self.get_tflite_results(inputs_dict, model_path)

    def get_tf2_keras_results(self, inputs_dict, model_path):
        import tensorflow as tf
        import numpy as np

        # load a model
        loaded_model = tf.keras.models.load_model(model_path, custom_objects=None)

        # prepare input
        input_for_model = []

        # order inputs based on input names in tests
        # since TF2 Keras model accepts a list of tensors for prediction
        for input_name in sorted(inputs_dict):
            input_value = inputs_dict[input_name]
            # convert NCHW to NHWC layout of tensor rank greater 3
            if self.use_old_api:
                if len(input_value.shape) == 4:
                    input_value = np.transpose(input_value, (0, 2, 3, 1))
                elif len(input_value.shape) == 5:
                    input_value = np.transpose(input_value, (0, 2, 3, 4, 1))
            input_for_model.append(input_value)
        if len(input_for_model) == 1:
            input_for_model = input_for_model[0]

        # infer by original framework and complete a dictionary with reference results
        tf_res_list = loaded_model(input_for_model)
        if tf.is_tensor(tf_res_list):
            tf_res_list = [tf_res_list]
        else:
            # in this case tf_res_list is a list of the single tuple of outputs
            tf_res_list = tf_res_list[0]

        result = dict()
        for ind, tf_res in enumerate(tf_res_list):
            if ind == 0:
                output = "Identity"
            else:
                output = "Identity_{}".format(ind)

            tf_res = tf_res.numpy()
            tf_res_rank = len(tf_res.shape)
            if self.use_old_api and tf_res_rank > 3:
                # create axis order for NCHW layout
                axis_order = np.arange(tf_res_rank)
                axis_order = np.insert(axis_order, 1, axis_order[-1])
                axis_order = np.delete(axis_order, [-1])
                result[output] = tf_res.transpose(axis_order)
            else:
                result[output] = tf_res
        return result

    def get_tflite_results(self, inputs_dict, model_path):
        import tensorflow as tf
        interpreter = tf.compat.v1.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_name_to_id_mapping = {input['name']: input['index'] for input in input_details}

        for layer, data in inputs_dict.items():
            tensor_index = input_name_to_id_mapping[layer]
            tensor_id = next(i for i, tensor in enumerate(input_details) if tensor['index'] == tensor_index)
            interpreter.set_tensor(input_details[tensor_id]['index'], data)

        interpreter.invoke()
        tf_result = dict()
        for output in output_details:
            tf_result[output['name']] = interpreter.get_tensor(output['index'])

        result = dict()
        for out in tf_result.keys():
            _tf_res = tf_result[out]
            result[out] = transpose_nhwc_to_nchw(_tf_res, self.use_new_frontend,
                                                 self.use_old_api)

        return tf_result
