# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

from common.layer_test_class import CommonLayerTest
from common.utils.tf_utils import summarize_graph


def transpose_nchw_to_nhwc(data, use_new_frontend, use_old_api):
    if use_new_frontend or not use_old_api:
        return data

    if len(data.shape) == 4:  # reshaping for 4D tensors
        return data.transpose(0, 2, 3, 1)
    elif len(data.shape) == 5:  # reshaping for 5D tensors
        return data.transpose(0, 2, 3, 4, 1)
    else:
        return data


def transpose_nhwc_to_nchw(data, use_new_frontend, use_old_api):
    if use_new_frontend or not use_old_api:
        return data

    if len(data.shape) == 4:  # reshaping for 4D tensors
        return data.transpose(0, 3, 1, 2)  # 2, 0, 1
    elif len(data.shape) == 5:  # reshaping for 5D tensors
        return data.transpose(0, 4, 1, 2, 3)  # 3, 0, 1, 2
    else:
        return data


def save_to_pb(tf_model, path_to_saved_tf_model):
    import tensorflow as tf
    tf.io.write_graph(tf_model, path_to_saved_tf_model, 'model.pb', False)
    assert os.path.isfile(os.path.join(path_to_saved_tf_model, 'model.pb')), "model.pb haven't been saved " \
                                                                             "here: {}".format(path_to_saved_tf_model)
    return os.path.join(path_to_saved_tf_model, 'model.pb')


def save_pb_to_tflite(pb_model):
    import tensorflow as tf

    graph_summary = summarize_graph(pb_model)
    inputs = [k for k in graph_summary['inputs'].keys()]
    outputs = graph_summary['outputs']

    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(pb_model, inputs, outputs)
    tflite_model = converter.convert()

    tflite_model_path = os.path.join(os.path.dirname(pb_model), 'model.tflite')
    with tf.io.gfile.GFile(tflite_model_path, 'wb') as f:
        f.write(tflite_model)

    return tflite_model_path


class CommonTFLayerTest(CommonLayerTest):
    def prepare_tf_inputs(self, inputs_dict):
        input = dict()
        for key in inputs_dict.keys():
            data = inputs_dict.get(key)
            if not ':' in key:
                key += ':0'
            input[key] = transpose_nchw_to_nhwc(data, self.use_new_frontend, self.use_old_api)

        return input

    def produce_model_path(self, framework_model, save_path):
        if not getattr(self, 'tflite', False):
            return save_to_pb(framework_model, save_path)
        else:
            pb_model = save_to_pb(framework_model, save_path)
            return save_pb_to_tflite(pb_model)

    def get_tf_results(self, inputs_dict, model_path):
        import tensorflow as tf
        from tensorflow.python.platform import gfile

        graph_summary = summarize_graph(model_path=model_path)
        outputs_list = graph_summary["outputs"]

        tf.compat.v1.reset_default_graph()

        with tf.compat.v1.Session() as sess:
            with gfile.FastGFile(model_path, 'rb') as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
                sess.graph.as_default()
                tf.compat.v1.import_graph_def(graph_def, name='')

                tf_res = sess.run([out + ":0" for out in outputs_list], inputs_dict)

                result = dict()
                for i, output in enumerate(outputs_list):
                    _tf_res = tf_res[i]
                    result[output] = transpose_nhwc_to_nchw(_tf_res, self.use_new_frontend,
                                                            self.use_old_api)
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

    def get_framework_results(self, inputs_dict, model_path):
        if not getattr(self, 'tflite', False):
            # prepare inputs
            inputs_dict = self.prepare_tf_inputs(inputs_dict)
            # get results from tensorflow
            return self.get_tf_results(inputs_dict, model_path)
        else:
            # get results from tflite
            return self.get_tflite_results(inputs_dict, model_path)
