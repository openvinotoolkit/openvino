# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

from common.layer_test_class import CommonLayerTest
from common.utils.tflite_utils import get_tflite_results, save_tf2_saved_model_to_tflite


def save_to_tf2_savedmodel(tf2_model, path_to_saved_tf2_model):
    import tensorflow as tf
    assert int(tf.__version__.split('.')[0]) >= 2, "TensorFlow 2 must be used for this suite validation"
    tf.keras.models.save_model(tf2_model, path_to_saved_tf2_model, save_format='tf')
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
            return get_tflite_results(self.use_new_frontend, self.use_old_api, inputs_dict, model_path)

    def get_tf2_keras_results(self, inputs_dict, model_path):
        import tensorflow as tf
        imported = tf.saved_model.load(model_path)
        f = imported.signatures["serving_default"]
        tf_outputs = f(**inputs_dict)
        return tf_outputs
