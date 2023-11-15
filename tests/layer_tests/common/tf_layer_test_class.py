# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from common.layer_test_class import CommonLayerTest
from common.utils.tf_utils import summarize_graph

from common.utils.tflite_utils import get_tflite_results, save_pb_to_tflite
from common.utils.tf_utils import save_to_pb, transpose_nhwc_to_nchw, transpose_nchw_to_nhwc


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
        outputs_list = [out + ":0" for out in outputs_list]

        tf.compat.v1.reset_default_graph()

        with tf.compat.v1.Session() as sess:
            with gfile.FastGFile(model_path, 'rb') as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
                sess.graph.as_default()
                tf.compat.v1.import_graph_def(graph_def, name='')

                tf_res = sess.run(outputs_list, inputs_dict)

                result = dict()
                for i, output in enumerate(outputs_list):
                    _tf_res = tf_res[i]
                    result[output] = transpose_nhwc_to_nchw(_tf_res, self.use_new_frontend,
                                                            self.use_old_api)
                return result

    def get_framework_results(self, inputs_dict, model_path):
        if not getattr(self, 'tflite', False):
            # prepare inputs
            inputs_dict = self.prepare_tf_inputs(inputs_dict)
            # get results from tensorflow
            return self.get_tf_results(inputs_dict, model_path)
        else:
            # get results from tflite
            return get_tflite_results(self.use_new_frontend, self.use_old_api, inputs_dict, model_path)
