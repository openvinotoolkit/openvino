# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import os
import shutil

import gc
import tempfile
import pytest
import tensorflow as tf
import openvino as ov
import tensorflow_hub as hub
# noinspection PyUnresolvedReferences
import numpy as np

from models_hub_common.test_performance_model import TestModelPerformance
from models_hub_common.utils import get_models_list
from models_hub_common.constants import tf_hub_cache_dir
from models_hub_common.constants import no_clean_cache_dir


def clean_cache():
    if not os.path.exists(tf_hub_cache_dir):
        return
    for file_name in os.listdir(tf_hub_cache_dir):
        file_path = os.path.join(tf_hub_cache_dir, file_name)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            pass


class TestTFPerformanceModel(TestModelPerformance):
    def load_model(self, model_name, model_link):
        hub.load(model_link)
        return hub.resolve(model_link)

    def get_inputs_info(self, model_path: str):
        inputs_info = []
        core = ov.Core()
        model = core.read_model(model=model_path)
        for param in model.inputs:
            input_shape = []
            param_shape = param.get_node().get_output_partial_shape(0)
            shape_special_dims = [ov.Dimension(), ov.Dimension(), ov.Dimension(), ov.Dimension(3)]
            if param_shape == ov.PartialShape(shape_special_dims) and param.get_element_type() == ov.Type.f32:
                # image classification case, let us imitate an image
                # that helps to avoid compute output size issue
                input_shape = [1, 200, 200, 3]
            else:
                for dim in param_shape:
                    if dim.is_dynamic:
                        input_shape.append(1)
                    else:
                        input_shape.append(dim.get_length())
            inputs_info.append((param.get_node().get_friendly_name(), input_shape, param.get_element_type()))
        return inputs_info

    def get_converted_model(self, model_path: str):
        return ov.convert_model(model_path)

    def get_read_model(self, model_path: str):
        core = ov.Core()
        return core.read_model(model=model_path)

    def teardown_method(self):
        if not no_clean_cache_dir:
            clean_cache()
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

