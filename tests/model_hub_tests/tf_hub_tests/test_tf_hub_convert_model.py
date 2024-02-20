# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import gc
import os
import shutil
import subprocess
import tempfile

import pytest
import tensorflow as tf
import tensorflow.compat.v1 as tf_v1
import tensorflow_hub as hub
# noinspection PyUnresolvedReferences
import tensorflow_text  # do not delete, needed for text models
from models_hub_common.constants import tf_hub_cache_dir
from models_hub_common.test_convert_model import TestConvertModel
from models_hub_common.utils import get_models_list
from tf_hub_tests.utils import type_map, load_graph, get_input_signature, get_output_signature


class TestTFHubConvertModel(TestConvertModel):
    def setup_class(self):
        self.model_dir = tempfile.TemporaryDirectory()

    def load_model(self, model_name, model_link):
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

    def get_inputs_info(self, model_obj):
        if type(model_obj) is tf_v1.Graph:
            input_signature = get_input_signature(model_obj)
        else:
            assert len(model_obj.structured_input_signature) > 1, "incorrect model or test issue"
            input_signature = model_obj.structured_input_signature[1].items()

        inputs_info = []
        for input_name, input_info in input_signature:
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

        output_dict = {}
        for out_name, out_value in model_obj(**tf_inputs).items():
            output_dict[out_name] = out_value.numpy()

        return output_dict

    def teardown_method(self):
        # remove all downloaded files for TF Hub models
        for file_name in os.listdir(tf_hub_cache_dir):
            file_path = os.path.join(tf_hub_cache_dir, file_name)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                pass
        # deallocate memory after each test case
        gc.collect()

    @pytest.mark.parametrize("model_name,model_link,mark,reason",
                             get_models_list(os.path.join(os.path.dirname(__file__), "precommit_models")))
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
                             get_models_list(os.path.join(os.path.dirname(__file__), "nightly_models")))
    @pytest.mark.nightly
    def test_convert_model_all_models(self, model_name, model_link, mark, reason, ie_device):
        assert mark is None or mark == 'skip' or mark == 'xfail', \
            "Incorrect test case: {}, {}".format(model_name, model_link)
        if mark == 'skip':
            pytest.skip(reason)
        elif mark == 'xfail':
            pytest.xfail(reason)
        self.run(model_name, model_link, ie_device)
