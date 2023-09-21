# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import gc
import os
import shutil

import pytest
import tensorflow as tf
import tensorflow_hub as hub
# noinspection PyUnresolvedReferences
import tensorflow_text  # do not delete, needed for text models
from models_hub_common.constants import tf_hub_cache_dir
from models_hub_common.test_convert_model import TestConvertModel
from models_hub_common.utils import get_models_list
from tf_hub_tests.utils import type_map


class TestTFHubConvertModel(TestConvertModel):
    def load_model(self, model_name, model_link):
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
        inputs_info = []
        assert len(model_obj.structured_input_signature) > 1, "incorrect model or test issue"
        for input_name, input_info in model_obj.structured_input_signature[1].items():
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
        assert mark is None or mark == 'skip', "Incorrect test case: {}, {}".format(model_name, model_link)
        if mark == 'skip':
            pytest.skip(reason)
        self.run(model_name, model_link, ie_device)

    @pytest.mark.parametrize("model_name,model_link,mark,reason",
                             get_models_list(os.path.join(os.path.dirname(__file__), "nightly_models")))
    @pytest.mark.nightly
    def test_convert_model_all_models(self, model_name, model_link, mark, reason, ie_device):
        assert mark is None or mark == 'skip', "Incorrect test case: {}, {}".format(model_name, model_link)
        if mark == 'skip':
            pytest.skip(reason)
        self.run(model_name, model_link, ie_device)
