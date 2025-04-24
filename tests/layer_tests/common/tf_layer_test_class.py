# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from common.layer_utils import InferAPI
from common.utils.common_utils import generate_ir_python_api
from common.utils.tf_utils import save_to_pb
from common.utils.tf_utils import summarize_graph
from common.utils.tflite_utils import get_tflite_results, save_pb_to_tflite
from pathlib import Path


class CommonTFLayerTest:
    input_model_key = "input_model"

    def prepare_tf_inputs(self, inputs_dict):
        input = dict()
        for key in inputs_dict.keys():
            data = inputs_dict.get(key)
            if not ':' in key:
                key += ':0'
            input[key] = data

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
        fw_outputs_list = [out + ":0" for out in outputs_list]
        if not self.use_legacy_frontend:
            outputs_list = fw_outputs_list

        tf.compat.v1.reset_default_graph()

        with tf.compat.v1.Session() as sess:
            with gfile.FastGFile(model_path, 'rb') as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
                sess.graph.as_default()
                tf.compat.v1.import_graph_def(graph_def, name='')

                tf_res = sess.run(fw_outputs_list, inputs_dict)

                result = dict()
                for i, output in enumerate(outputs_list):
                    _tf_res = tf_res[i]
                    result[output] = _tf_res
                return result

    def get_framework_results(self, inputs_dict, model_path):
        if not getattr(self, 'tflite', False):
            # prepare inputs
            inputs_dict = self.prepare_tf_inputs(inputs_dict)
            # get results from tensorflow
            return self.get_tf_results(inputs_dict, model_path)
        else:
            # get results from tflite
            return get_tflite_results(self.use_legacy_frontend, inputs_dict, model_path)

    def _test(self, framework_model, ref_net, ie_device, precision, ir_version, temp_dir,
              use_legacy_frontend=False, infer_timeout=60, **kwargs):
        model_path = self.produce_model_path(framework_model=framework_model, save_path=temp_dir)
        self.use_legacy_frontend = use_legacy_frontend

        compress_to_fp16 = False if precision == 'FP32' else True

        if use_legacy_frontend:
            mo_params = {self.input_model_key: model_path,
                         "output_dir": temp_dir,
                         "compress_to_fp16": compress_to_fp16,
                         "model_name": 'model'}

            if 'input_shapes' in kwargs and len(kwargs['input_shapes']):
                input_shapes_str = []
                for ishape in kwargs['input_shapes']:
                    input_shapes_str.append('[' + ','.join([str(i) for i in ishape]) + ']')
                mo_params.update(dict(input_shape=','.join(input_shapes_str)))

            if 'input_names' in kwargs and len(kwargs['input_names']):
                mo_params.update(dict(input=','.join(kwargs['input_names'])))
            mo_params["use_legacy_frontend"] = True
        else:
            # pack input parameters for convert_model of OVC
            # that are different from MO
            mo_params = {"input_model": model_path,
                         "output_dir": temp_dir,
                         "compress_to_fp16": compress_to_fp16
                         }

            if 'input_shapes' in kwargs and 'input_names' in kwargs:
                input_shapes = kwargs['input_shapes']
                input_names = kwargs['input_names']
                assert len(input_shapes) == len(input_names)
                input_dict = {}
                for input_name, input_shape in zip(input_names, input_shapes):
                    input_dict[input_name] = input_shape
                mo_params.update(dict(input=input_dict))
            elif 'input_names' in kwargs:
                mo_params.update(dict(input=kwargs['input_names']))
            elif 'input_shapes' in kwargs:
                mo_params.update(dict(input=kwargs['input_shapes']))

        exit_code, stderr = generate_ir_python_api(**mo_params)
        assert not exit_code, (
            "IR generation failed with {} exit code: {}".format(exit_code, stderr))

        path_to_xml = Path(temp_dir, 'model.xml')
        path_to_bin = Path(temp_dir, 'model.bin')

        config = None
        # GPU default execution precision is FP16, so if we want to check FP32 inference
        # we need to set explicit precision hint
        if ie_device == 'GPU' and precision == 'FP32':
            config = {'INFERENCE_PRECISION_HINT': 'f32'}

        ie_engine = InferAPI(model=path_to_xml,
                             weights=path_to_bin,
                             device=ie_device,
                             use_legacy_frontend=use_legacy_frontend)
        # Prepare feed dict
        if 'kwargs_to_prepare_input' in kwargs and kwargs['kwargs_to_prepare_input']:
            inputs_dict = self._prepare_input(ie_engine.get_inputs_info(precision),
                                              kwargs['kwargs_to_prepare_input'])
        else:
            inputs_dict = self._prepare_input(ie_engine.get_inputs_info(precision))

        # Infer using OpenVINO
        infer_res = ie_engine.infer(input_data=inputs_dict, infer_timeout=infer_timeout, config=config)

        # Infer using the original framework (TensorFlow or TFLite)
        fw_res = self.get_framework_results(inputs_dict=inputs_dict, model_path=model_path)

        if 'custom_eps' in kwargs and kwargs['custom_eps'] is not None:
            custom_eps = kwargs['custom_eps']
        else:
            if precision == 'FP32':
                custom_eps = 1e-4
            else:
                custom_eps = 5e-2
        # Compare Ie results with Framework results
        assert self.compare_ie_results_with_framework(infer_res=infer_res, framework_res=fw_res,
                                                      framework_eps=custom_eps), \
            "Comparing with Framework failed: ie_res={}; framework_res={}.".format(infer_res,
                                                                                   fw_res)

    # Feed dict for each input is filled with random number.
    # It is possible to redefine this function and generate your own input
    def _prepare_input(self, inputs_dict):
        for input in inputs_dict.keys():
            inputs_dict[input] = np.random.randint(-10, 10, inputs_dict[input]).astype(np.float32)
        return inputs_dict

    def compare_ie_results_with_framework(self, infer_res, framework_res, framework_eps):
        is_ok = True
        from common.utils.common_utils import allclose
        for framework_out_name in framework_res:
            if framework_out_name not in infer_res and len(infer_res) == 1:
                ov_res = list(infer_res.values())[0]
            else:
                ov_res = infer_res[framework_out_name]

            fw_res = np.array(framework_res[framework_out_name])

            assert fw_res.dtype == ov_res.dtype or \
                   ov_res.dtype.type == str or \
                   ov_res.dtype.type == np.str_, 'Outputs types are different: ' \
                                                 'OpenVINO output type - {}, ' \
                                                 'Framework output type - {}'.format(ov_res.dtype, fw_res.dtype)
            assert fw_res.shape == ov_res.shape, 'Outputs shapes are different: ' \
                                                 'OpenVINO output shape - {}, ' \
                                                 'Framework output shape - {}'.format(ov_res.shape, fw_res.shape)

            if not allclose(ov_res, fw_res,
                            atol=framework_eps,
                            rtol=framework_eps):
                is_ok = False
                if ov_res.dtype != bool:
                    diff = np.array(abs(ov_res - fw_res)).max()
                    print("Max diff is {}".format(diff))
                else:
                    print("Boolean results are not equal")
            else:
                print("Accuracy validation successful!\n")
                print("absolute eps: {}, relative eps: {}".format(framework_eps, framework_eps))
        return is_ok
