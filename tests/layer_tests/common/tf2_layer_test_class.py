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
            if len(input_value.shape) == 4:
                input_value = np.transpose(input_value, (0, 2, 3, 1))
            elif len(input_value.shape) == 5:
                input_value = np.transpose(input_value, (0, 2, 3, 4, 1))
            input_for_model.append(input_value)
        if len(input_for_model) == 1:
            input_for_model = input_for_model[0]

        # prepare output
        # TODO: compute output layer names acoording to a graph
        # Now it supports only single output layer that called as "Identity" always
        output_layers = ["Identity"]

        # infer by original framework
        result = dict()
        for i, output in enumerate(output_layers):
            # TODO: infer result for each output_layer
            _tf_res = loaded_model(input_for_model).numpy()
            if len(_tf_res.shape) == 4:  # reshaping for 4D tensors
                result[output] = _tf_res.transpose(0, 3, 1, 2)  # 2, 0, 1
            elif len(_tf_res.shape) == 5:  # reshaping for 5D tensors
                result[output] = _tf_res.transpose(0, 4, 1, 2, 3)  # 3, 0, 1, 2
            else:
                result[output] = _tf_res
        return result
