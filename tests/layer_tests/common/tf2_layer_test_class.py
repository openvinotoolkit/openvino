# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

from common.layer_test_class import CommonLayerTest


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
        return save_to_tf2_savedmodel(framework_model, save_path)

    def get_framework_results(self, inputs_dict, model_path):
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
