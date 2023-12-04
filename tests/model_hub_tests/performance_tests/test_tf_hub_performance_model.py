# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import gc
import os
import shutil

import pytest
import tensorflow_hub as hub
from models_hub_common.constants import no_clean_cache_dir
from models_hub_common.constants import tf_hub_cache_dir
from models_hub_common.test_performance_model import TestModelPerformance
from models_hub_common.utils import get_models_list


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
        except Exception:
            pass


class TestTFPerformanceModel(TestModelPerformance):
    def load_model(self, model_name, model_link):
        hub.load(model_link)
        return hub.resolve(model_link)

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
