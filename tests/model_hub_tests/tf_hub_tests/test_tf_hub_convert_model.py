# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_hub as hub
# noinspection PyUnresolvedReferences
import tensorflow_text  # do not delete, needed for text models
from models_hub_common.test_convert_model import TestConvertModel
from models_hub_common.utils import get_models_list


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
        for input_info in model_obj.inputs:
            input_shape = []
            try:
                for dim in input_info.shape.as_list():
                    if dim is None:
                        input_shape.append(1)
                    else:
                        input_shape.append(dim)
            except ValueError:
                # unknown rank case
                pass
            type_map = {
                tf.float64: np.float64,
                tf.float32: np.float32,
                tf.int8: np.int8,
                tf.int16: np.int16,
                tf.int32: np.int32,
                tf.int64: np.int64,
                tf.uint8: np.uint8,
                tf.uint16: np.uint16,
                tf.string: str,
                tf.bool: bool,
            }
            if input_info.dtype not in type_map:
                continue
            assert input_info.dtype in type_map, "Unsupported input type: {}".format(input_info.dtype)
            inputs_info.append((input_shape, type_map[input_info.dtype]))

        return inputs_info

    def infer_fw_model(self, model_obj, inputs):
        # TODO 119141 - use the same dictionary for OV inference
        tf_inputs = {}
        for input_ind, input_name in enumerate(sorted(model_obj.structured_input_signature[1].keys())):
            tf_inputs[input_name] = tf.constant(inputs[input_ind])

        output_dict = {}
        for out_name, out_value in model_obj(**tf_inputs).items():
            output_dict[out_name] = out_value.numpy()

        # TODO: 119141 - remove this workaround
        # map external tensor names to internal names
        assert len(model_obj.outputs) == len(model_obj.structured_outputs)
        fw_outputs = {}
        for output_ind, external_name in enumerate(sorted(model_obj.structured_outputs.keys())):
            internal_name = model_obj.outputs[output_ind].name
            out_value = output_dict[external_name]
            fw_outputs[internal_name] = out_value
        return fw_outputs

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
