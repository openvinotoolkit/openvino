# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import gc
import os
import shutil
from collections import namedtuple
from enum import Enum

import pytest
import tensorflow_hub as hub
from models_hub_common.test_performance_model import TestModelPerformance
import models_hub_common.utils as utils
import models_hub_common.constants as const


Conf = namedtuple("Conf", "runtime_measure_duration runtime_heat_duration")


class TestType(Enum):
    PRECOMMIT = 0
    NIGHTLY = 1


def get_tests_conf(test_type: TestType) -> Conf:
    options = {TestType.NIGHTLY:
               Conf(utils.nano_secs(const.nightly_runtime_measure_duration),
                    utils.nano_secs(const.nigtly_runtime_heat_duration)),
               TestType.PRECOMMIT:
               Conf(utils.nano_secs(const.precommit_runtime_measure_duration),
                    utils.nano_secs(const.precommit_runtime_heat_duration))}
    return options[test_type]


def clean_cache():
    if not os.path.exists(const.tf_hub_cache_dir):
        return
    for file_name in os.listdir(const.tf_hub_cache_dir):
        file_path = os.path.join(const.tf_hub_cache_dir, file_name)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception:
            pass


def get_nightly_config_path(config_name):
    dir_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tensorflow", "model_lists")
    return os.path.join(dir_path, config_name)


def get_local_config_path(config_name):
    return os.path.join(os.path.dirname(__file__), config_name)


class TestTFPerformanceModel(TestModelPerformance):
    def load_model(self, model_name, model_link):
        hub.load(model_link)
        return hub.resolve(model_link)

    def teardown_method(self):
        if not const.no_clean_cache_dir:
            utils.cleanup_dir(const.tf_hub_cache_dir)
        # deallocate memory after each test case
        gc.collect()

    @pytest.mark.parametrize("model_name,model_link,mark,reason",
                             utils.get_models_list(get_local_config_path("precommit_models")))
    @pytest.mark.precommit
    def test_convert_model_precommit(self, model_name, model_link, mark, reason, ie_device):
        assert mark is None or mark == 'skip', "Incorrect test case: {}, {}".format(model_name, model_link)
        if mark == 'skip':
            pytest.skip(reason)
        self.run(model_name, model_link, ie_device, get_tests_conf(TestType.PRECOMMIT))

    @pytest.mark.parametrize("model_name,model_link,mark,reason",
                             utils.get_models_list_not_skipped(get_nightly_config_path("nightly_tf_hub"),
                                                               get_local_config_path("nightly_models.skip")))
    @pytest.mark.nightly
    def test_convert_model_all_models(self, model_name, model_link, mark, reason, ie_device):
        assert mark is None or mark == 'skip', "Incorrect test case: {}, {}".format(model_name, model_link)
        if mark == 'skip':
            pytest.skip(reason)
        self.run(model_name, model_link, ie_device, get_tests_conf(TestType.NIGHTLY))
