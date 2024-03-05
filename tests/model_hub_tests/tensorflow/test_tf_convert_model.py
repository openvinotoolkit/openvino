# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import gc
import os
import shutil
import subprocess
import tempfile

import numpy as np
import pytest
import tensorflow as tf
import tensorflow.compat.v1 as tf_v1
import tensorflow_hub as hub
# noinspection PyUnresolvedReferences
import tensorflow_text  # do not delete, needed for text models

from models_hub_common.constants import tf_hub_cache_dir, hf_cache_dir
from models_hub_common.test_convert_model import TestConvertModel
from models_hub_common.utils import compare_two_tensors
from models_hub_common.utils import get_models_list
from utils import type_map, load_graph, get_input_signature, get_output_signature


def is_hf_link(link: str):
    return link.startswith("hf_")


def get_output_signature_from_keras_layer(model):
    try:
        from openvino.frontend.tensorflow.utils import trace_tf_model_if_needed
        traced_model = trace_tf_model_if_needed(model, None, None, None)
        return traced_model.structured_outputs
    except:
        return None


def tf_tensor_to_numpy(tensor):
    if isinstance(tensor, tf.Tensor):
        return tensor.numpy()
    elif isinstance(tensor, (list, tuple)):
        res = []
        for elem in tensor:
            res.append(tf_tensor_to_numpy(elem))
        if isinstance(tensor, list):
            return res
        else:
            return tuple(res)
    elif isinstance(tensor, dict):
        res = {}
        for out_name, out_value in tensor.items():
            res[out_name] = tf_tensor_to_numpy(out_value)
        return res
    raise Exception("Unknown output type of original FW inference result: {}".format(type(tensor)))


def compare_by_signature(ov_out, ref, signature, fw_eps):
    if isinstance(ref, np.ndarray):
        assert isinstance(signature, tf.Tensor), "Accuracy comparison failed due to mismatch of obtained signature and FW outputs."
        return compare_two_tensors(ov_out[signature.name], ref, fw_eps)
    elif isinstance(ref, (list, tuple)):
        res = True
        for idx, elem in enumerate(ref):
            res = res and compare_by_signature(ov_out, elem, signature[idx], fw_eps)
        return res
    elif isinstance(ref, dict):
        res = True
        for out_name, out_value in ref.items():
            res = res and compare_by_signature(ov_out, out_value, signature[out_name], fw_eps)
        return res
    raise Exception("Unknown FW reference type: {}".format(type(ref)))


class TestTFHubConvertModel(TestConvertModel):
    def setup_class(self):
        self.model_dir = tempfile.TemporaryDirectory()

    def load_model(self, model_name, model_link: str):
        if is_hf_link(model_link):
            library_type = model_link[3:]
            if library_type == "transformers":
                from transformers import TFAutoModel
                return TFAutoModel.from_pretrained(model_name)
            elif library_type == "sentence-transformers":
                from tf_sentence_transformers import SentenceTransformer
                return SentenceTransformer.from_pretrained(model_name)
        if 'storage.openvinotoolkit.org' in model_link:
            # this models is from public OpenVINO storage
            subprocess.check_call(["wget", "-nv", model_link], cwd=self.model_dir.name)
            model_file_name = os.path.basename(model_link)
            if model_file_name.endswith('.tar.gz'):
                # unzip archive and try to find the frozen model
                subprocess.check_call(["tar", "-xvzf", model_file_name], cwd=self.model_dir.name)
                model_file_name = os.path.join(self.model_dir.name, model_file_name[:-7], 'frozen_inference_graph.pb')
            else:
                model_file_name = os.path.join(self.model_dir.name, model_file_name)
            if model_file_name.endswith('.pb'):
                graph = load_graph(model_file_name)
                return graph

        load = hub.load(model_link)
        if 'serving_default' in list(load.signatures.keys()):
            concrete_func = load.signatures['serving_default']
        elif 'default' in list(load.signatures.keys()):
            concrete_func = load.signatures['default']
        else:
            signature_keys = sorted(list(load.signatures.keys()))
            assert len(signature_keys) > 0, "No signatures for a model {}, url {}".format(model_name, model_link)
            concrete_func = load.signatures[signature_keys[0]]
        concrete_func._backref_to_saved_model = load
        return concrete_func

    def compare_results(self, fw_outputs, ov_outputs):
        fw_eps = 5e-2

        # TF FE loses output structure in case when original model has output dictionary, where values are tuples.
        # OV generates in this case a list of tensors with inner tensor names.
        # Comparison of results in this case can be performed if output signature is known.
        # TODO: When OV supports output dictionary of tuples need to remove usage of output signature here and update comparison method - Ticket TODO
        if len(fw_outputs) != len(ov_outputs):
            assert self.output_signature is not None, "The number of outputs between framework and OpenVINO is different and could not obtain output signature."
            return compare_by_signature(ov_outputs, fw_outputs, self.output_signature, fw_eps)

        is_ok = True
        if isinstance(fw_outputs, dict):
            for out_name in fw_outputs.keys():
                cur_fw_res = fw_outputs[out_name]
                assert out_name in ov_outputs, \
                    "OpenVINO outputs does not contain tensor with name {}".format(out_name)
                cur_ov_res = ov_outputs[out_name]
                print(f"fw_re: {cur_fw_res};\n ov_res: {cur_ov_res}")
                is_ok = is_ok and compare_two_tensors(cur_ov_res, cur_fw_res, fw_eps)
        else:
            for i in range(len(ov_outputs)):
                cur_fw_res = fw_outputs[i]
                cur_ov_res = ov_outputs[i]
                print(f"fw_res: {cur_fw_res};\n ov_res: {cur_ov_res}")
                is_ok = is_ok and compare_two_tensors(cur_ov_res, cur_fw_res, fw_eps)
        assert is_ok, "Accuracy validation failed"

    def get_inputs_info(self, model_obj):
        if type(model_obj) is tf_v1.Graph:
            input_signature = get_input_signature(model_obj)
        elif hasattr(model_obj, "input_signature") and model_obj.input_signature is not None:
            input_signature = model_obj.input_signature
        else:
            assert len(model_obj.structured_input_signature) > 1, "incorrect model or test issue"
            input_signature = model_obj.structured_input_signature[1].items()

        inputs_info = []
        for input_name, input_info in (input_signature.items() if isinstance(input_signature, dict) else input_signature):
            input_shape = []
            try:
                if input_info.shape.as_list() == [None, None, None, 3] and input_info.dtype == tf.float32:
                    # image classification case, let us imitate an image
                    # that helps to avoid compute output size issue
                    input_shape = [1, 200, 200, 3]
                else:
                    for dim in input_info.shape.as_list():
                        if dim is None:
                            input_shape.append(1)
                        else:
                            input_shape.append(dim)
            except ValueError:
                # unknown rank case
                pass
            if input_info.dtype == tf.resource:
                # skip inputs corresponding to variables
                continue
            assert input_info.dtype in type_map, "Unsupported input type: {}".format(input_info.dtype)
            inputs_info.append((input_name, input_shape, type_map[input_info.dtype]))

        self.output_signature = get_output_signature_from_keras_layer(model_obj)
        return inputs_info

    def infer_fw_model(self, model_obj, inputs):
        if type(model_obj) is tf_v1.Graph:
            # infer tf.Graph object
            feed_dict = {}
            for input_name, input_value in inputs.items():
                input_tensor = model_obj.get_tensor_by_name(input_name)
                feed_dict[input_tensor] = input_value

            # compute output signature
            output_names = get_output_signature(model_obj)
            outputs = []
            for output_name in output_names:
                outputs.append(model_obj.get_tensor_by_name(output_name))

            with tf_v1.Session(graph=model_obj) as sess:
                tf_output = sess.run(outputs, feed_dict=feed_dict)

            output_dict = {}
            for ind, output_name in enumerate(output_names):
                output_dict[output_name] = tf_output[ind]
            return output_dict

        # repack input dictionary to tensorflow constants
        tf_inputs = {}
        for input_name, input_value in inputs.items():
            tf_inputs[input_name] = tf.constant(input_value)
        return tf_tensor_to_numpy(model_obj(**tf_inputs))

    def clean_dir(self, dir_name: str):
        if os.path.exists(dir_name):
            for file_name in os.listdir(dir_name):
                file_path = os.path.join(dir_name, file_name)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    pass

    def teardown_method(self):
        # remove all downloaded files for TF Hub models
        self.clean_dir(tf_hub_cache_dir)

        # remove all downloaded files for HF models
        self.clean_dir(hf_cache_dir)

        # deallocate memory after each test case
        gc.collect()

    @pytest.mark.parametrize("model_name,model_link,mark,reason",
                             get_models_list(os.path.join(os.path.dirname(__file__), "model_lists", "precommit")))
    @pytest.mark.precommit
    def test_convert_model_precommit(self, model_name, model_link, mark, reason, ie_device):
        assert mark is None or mark == 'skip' or mark == 'xfail', \
            "Incorrect test case: {}, {}".format(model_name, model_link)
        if mark == 'skip':
            pytest.skip(reason)
        elif mark == 'xfail':
            pytest.xfail(reason)
        self.run(model_name, model_link, ie_device)

    @pytest.mark.parametrize("model_name,model_link,mark,reason",
                             get_models_list(os.path.join(os.path.dirname(__file__), "model_lists", "nightly_tf_hub")))
    @pytest.mark.nightly_tf_hub
    def test_convert_model_all_models(self, model_name, model_link, mark, reason, ie_device):
        assert mark is None or mark == 'skip' or mark == 'xfail', \
            "Incorrect test case: {}, {}".format(model_name, model_link)
        if mark == 'skip':
            pytest.skip(reason)
        elif mark == 'xfail':
            pytest.xfail(reason)
        self.run(model_name, model_link, ie_device)

    @pytest.mark.parametrize("model_name,model_link,mark,reason",
                             get_models_list(os.path.join(os.path.dirname(__file__), "model_lists", "nightly_hf")))
    @pytest.mark.nightly_hf
    def test_convert_model_hugging_face_nightly(self, model_name, model_link, mark, reason, ie_device):
        assert mark is None or mark == 'skip' or mark == 'xfail', \
            "Incorrect test case: {}, {}".format(model_name, model_link)
        if mark == 'skip':
            pytest.skip(reason)
        elif mark == 'xfail':
            pytest.xfail(reason)
        self.run(model_name, model_link, ie_device)
