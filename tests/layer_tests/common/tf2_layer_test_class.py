# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

from common.layer_test_class import CommonLayerTest
from common.utils.tflite_utils import get_tflite_results, save_tf2_saved_model_to_tflite


def save_to_tf2_savedmodel(tf2_model, path_to_saved_tf2_model):
    import tensorflow as tf
    assert int(tf.__version__.split('.')[0]) >= 2, "TensorFlow 2 must be used for this suite validation"
    # Since TF 2.16 this is only way to serialize Keras objects into SavedModel format
    tf2_model.export(path_to_saved_tf2_model)
    assert os.path.isdir(path_to_saved_tf2_model), "the model haven't been saved " \
                                                   "here: {}".format(path_to_saved_tf2_model)
    return path_to_saved_tf2_model


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
            return get_tflite_results(self.use_legacy_frontend, inputs_dict, model_path)

    def get_tf2_keras_results(self, inputs_dict, model_path):
        import tensorflow as tf

        result = dict()
        if not self.use_legacy_frontend:
            imported = tf.saved_model.load(model_path)
            f = imported.signatures["serving_default"]
            result = f(**inputs_dict)
        else:
            # load a model
            loaded_model = tf.keras.models.load_model(model_path, custom_objects=None)
            # prepare input
            input_for_model = []
            # order inputs based on input names in tests
            # since TF2 Keras model accepts a list of tensors for prediction
            for input_name in sorted(inputs_dict):
                input_value = inputs_dict[input_name]
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

            for ind, tf_res in enumerate(tf_res_list):
                if ind == 0:
                    output = "Identity"
                else:
                    output = "Identity_{}".format(ind)
                tf_res = tf_res.numpy()
                result[output] = tf_res

        return result
