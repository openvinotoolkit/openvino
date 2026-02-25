# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import gc
import os
import shutil

import pytest
import tensorflow as tf
import tensorflow_hub as hub
# noinspection PyUnresolvedReferences
import tensorflow_text  # do not delete, needed for text models
from models_hub_common.test_convert_model import TestConvertModel
from models_hub_common.utils import get_models_list
from openvino import Core, PartialShape

from utils import unpack_tf_result, retrieve_inputs_info_for_signature


class TestTFReadModel(TestConvertModel):
    def _reshape_if_required(self, ov_model, inputs):
        # check if any input of dynamic rank
        # if yes, set a static rank
        if isinstance(inputs, dict):
            needs_reshape = False
            new_shapes_dict = {}
            for model_input in ov_model.inputs:
                input_names = list(model_input.names)
                assert len(input_names) > 0, 'Expected at least one input tensor name'
                input_name = input_names[0]
                assert input_name in inputs, 'Inputs data does not contain {}'.format(input_name)
                input_shape = list(inputs[input_name].shape)
                new_shapes_dict[input_name] = PartialShape(input_shape)
                if model_input.get_partial_shape().rank.is_dynamic:
                    needs_reshape = True
            if needs_reshape:
                ov_model.reshape(new_shapes_dict)

    def _load_model_to_memory(self, model_path):
        # try the first attempt with default tag
        # ideally, to get the whole list of tags, it needs to use cli tool `saved_model_cli`
        try:
            return tf.saved_model.load(model_path)
        except:
            pass
        # try the second attempt with set() tag
        try:
            return tf.saved_model.load(model_path, tags=set())
        except:
            pass
        return None

    def load_model(self, _, model_link: str):
        # inference model from a disk
        # so use model path for both TF and OV inferences
        model_path = hub.resolve(model_link)
        return model_path

    def get_inputs_info(self, model_path):
        # load model into memory and retrieve inputs info (shape and type for each input)
        tf_model = self._load_model_to_memory(model_path)
        assert tf_model is not None, 'TensorFlow model is not loaded due to not found tag'
        inputs_info = []
        if 'serving_default' in list(tf_model.signatures.keys()):
            concrete_func = tf_model.signatures['serving_default']
            input_signature = concrete_func.structured_input_signature[1].items()
            inputs_info = retrieve_inputs_info_for_signature(input_signature)
        else:
            for signature in list(tf_model.signatures.keys()):
                concrete_func = tf_model.signatures[signature]
                input_signature = concrete_func.structured_input_signature[1].items()
                for input_info in retrieve_inputs_info_for_signature(input_signature):
                    inputs_info.append(input_info)
        return inputs_info

    def convert_model(self, model_path):
        # no need to convert
        # read_model is used during inference
        return model_path

    def infer_ov_model(self, model_path, inputs, ie_device):
        core = Core()
        ov_model = core.read_model(model_path)
        self._reshape_if_required(ov_model, inputs)
        compiled = core.compile_model(ov_model, ie_device)
        ov_outputs = compiled(inputs)
        return ov_outputs

    def infer_fw_model(self, model_path, inputs):
        tf_model = self._load_model_to_memory(model_path)
        assert tf_model is not None, 'TensorFlow model is not loaded due to not found tag'
        if 'serving_default' in list(tf_model.signatures.keys()):
            concrete_func = tf_model.signatures['serving_default']
            # repack input dictionary to tensorflow constants
            tf_inputs = {}
            for input_name, input_value in inputs.items():
                tf_inputs[input_name] = tf.constant(input_value)
            return unpack_tf_result(concrete_func(**tf_inputs))
        else:
            output_results = {}
            for signature in list(tf_model.signatures.keys()):
                concrete_func = tf_model.signatures[signature]
                # repack input dictionary to tensorflow constants
                tf_inputs = {}
                for input_name in list(concrete_func.structured_input_signature[1].keys()):
                    tf_inputs[input_name] = tf.constant(inputs[input_name])

                output_tf_results = unpack_tf_result(concrete_func(**tf_inputs))
                assert isinstance(output_tf_results, dict), 'Expected dictionary output'

                for output_name, output_tensor in output_tf_results.items():
                    output_results[output_name] = output_tensor
            return output_results

    def _clean_dir(self, dir_name: str):
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
        # deallocate memory after each test case
        gc.collect()

    @pytest.mark.parametrize("model_name,model_link,mark,reason",
                             get_models_list(os.path.join(os.path.dirname(__file__),
                                                          "model_lists", "precommit_read_model")))
    @pytest.mark.precommit
    def test_read_model_precommit(self, model_name, model_link, mark, reason, ie_device):
        assert mark is None or mark == 'skip' or mark == 'xfail', \
            "Incorrect test case: {}, {}".format(model_name, model_link)
        if mark == 'skip':
            pytest.skip(reason)
        elif mark == 'xfail':
            pytest.xfail(reason)
        self.run(model_name, model_link, ie_device)
