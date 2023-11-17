# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import os
import shutil

import gc
import tarfile
import tempfile
import pytest
import tensorflow as tf
import openvino as ov
import tensorflow_hub as hub
# noinspection PyUnresolvedReferences
import tensorflow_text  # do not delete, needed for text models
import numpy as np
import json

from models_hub_common.test_performance_model import TestPerformanceModel
from models_hub_common.utils import get_models_list
from models_hub_common.constants import tf_hub_cache_dir
from models_hub_common.constants import no_clean_cache_dir


def clean_cache():
    for file_name in os.listdir(tf_hub_cache_dir):
        file_path = os.path.join(tf_hub_cache_dir, file_name)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            pass


class DownloadInfo:
    MODEL_LINK_KEY = 'model_link'
    PATH_KEY = 'path'

    def __init__(self, line: str = None):
        self.model_link = ''
        self.path = ''
        self.untar_path = ''
        if line:
            self.unpack(line)

    def unpack(self, s: str) -> None:
        item = json.loads(s)
        self.model_link = item[DownloadInfo.MODEL_LINK_KEY]
        self.path = item[DownloadInfo.PATH_KEY]

    def pack(self) -> str:
        item = dict()
        item[DownloadInfo.PATH_KEY] = self.path
        item[DownloadInfo.MODEL_LINK_KEY] = self.model_link
        return json.dumps(item)


DOWNLOADED_FILES_LIST_NAME = 'downloaded_files.txt'


def get_model_downloaded_info(model_link):
    path = os.path.join(tf_hub_cache_dir, DOWNLOADED_FILES_LIST_NAME)
    if not os.path.exists(path):
        return False
    try:
        file_lines = []
        with open(path, 'r') as f:
            file_lines = f.readlines()
        for line in file_lines:
            item = DownloadInfo(line)
            if item.model_link == model_link:
                return item
    except json.JSONDecodeError as e:
        print('downloaded files list is corrupted: remove cache')
        clean_cache()
        return None
    return None


def save_model_download_info(new_download_info):
    download_list_path = os.path.join(tf_hub_cache_dir, DOWNLOADED_FILES_LIST_NAME)
    with open(download_list_path, 'a') as f:
        f.write(new_download_info.pack())
        f.write('\n')


def download_model_from_tf_hub(model_name: str, model_link: str):
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
    return hub.resolve(model_link)


def get_model_list_path(filename: str) -> str:
    parent_dir = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(parent_dir, 'tf_hub_tests', filename)


class TestTFPerformanceModel(TestPerformanceModel):
    def load_model(self, model_name, model_link):
        downloaded_item = get_model_downloaded_info(model_link)
        model_path = None
        if downloaded_item and os.path.exists(downloaded_item.path):
            print('model is in tf hub cache')
            model_path = downloaded_item.path
        else:
            print('downloading model')
            model_path = download_model_from_tf_hub(model_name, model_link)
            if downloaded_item and not os.path.exists(downloaded_item.path):
                print('cache is broken - clean it')
                clean_cache()
            new_download_info = DownloadInfo()
            new_download_info.path = model_path
            new_download_info.model_link = model_link
            save_model_download_info(new_download_info)
        return model_path

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
                             get_models_list(get_model_list_path("precommit_models")))
    @pytest.mark.precommit
    def test_convert_model_precommit(self, model_name, model_link, mark, reason, ie_device):
        assert mark is None or mark == 'skip', "Incorrect test case: {}, {}".format(model_name, model_link)
        if mark == 'skip':
            pytest.skip(reason)
        self.run(model_name, model_link, ie_device)

    @pytest.mark.parametrize("model_name,model_link,mark,reason",
                             get_models_list(get_model_list_path("nightly_models")))
    @pytest.mark.nightly
    def test_convert_model_all_models(self, model_name, model_link, mark, reason, ie_device):
        assert mark is None or mark == 'skip', "Incorrect test case: {}, {}".format(model_name, model_link)
        if mark == 'skip':
            pytest.skip(reason)
        self.run(model_name, model_link, ie_device)
