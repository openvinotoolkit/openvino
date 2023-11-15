# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import os
import shutil

import gc
import wget
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
from models_hub_common.constants import wget_cache_dir
from models_hub_common.constants import no_clean_cache_dir


def unpack_tar_gz(path: str) -> str:
    parent_dir = os.path.dirname(path)
    target_dir = tempfile.mkdtemp(dir=parent_dir)
    with tarfile.open(path, 'r') as tar:
        tar.extractall(target_dir)
    return target_dir


class DownloadInfo:
    MODEL_LINK_KEY = 'model_link'
    PATH_KEY = 'path'
    UNTAR_PATH = 'untar_path'

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
        self.untar_path = item[DownloadInfo.UNTAR_PATH]

    def pack(self) -> str:
        item = dict()
        item[DownloadInfo.PATH_KEY] = self.path
        item[DownloadInfo.MODEL_LINK_KEY] = self.model_link
        item[DownloadInfo.UNTAR_PATH] = self.untar_path
        return json.dumps(item)


DOWNLOADED_FILES_LIST_NAME = 'downloaded_files.txt'


def clean_wget_cache():
    for file_name in os.listdir(wget_cache_dir):
        file_path = os.path.join(wget_cache_dir, file_name)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            pass


def get_model_downloaded_info(model_link):
    path = os.path.join(wget_cache_dir, DOWNLOADED_FILES_LIST_NAME)
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
        clean_wget_cache()
        return None
    return None


def save_model_download_info(new_download_info):
    download_list_path = os.path.join(wget_cache_dir, DOWNLOADED_FILES_LIST_NAME)
    with open(download_list_path, 'a') as f:
        f.write(new_download_info.pack())
        f.write('\n')


def download_model(model_name: str, model_link: str) -> str:
    downloaded_item = get_model_downloaded_info(model_link)
    model_path = None
    new_download_info = None
    if downloaded_item and os.path.exists(downloaded_item.path):
        model_path = downloaded_item.path
    else:
        # different downloaded files may have same names
        target_dir = tempfile.mkdtemp(dir=wget_cache_dir)
        model_path = os.path.join(wget_cache_dir, wget.download(model_link, out=target_dir))
        if downloaded_item and not os.path.exists(downloaded_item.path):
            clean_wget_cache()
        new_download_info = DownloadInfo()
        new_download_info.path = model_path
        new_download_info.model_link = model_link
    result_path = None
    if not tarfile.is_tarfile(model_path):
        result_path = model_path
    elif downloaded_item and os.path.exists(downloaded_item.untar_path):
        result_path = downloaded_item.untar_path
    else:
        result_path = unpack_tar_gz(model_path)
        if not new_download_info:
            new_download_info = DownloadInfo()
            new_download_info.path = model_path
            new_download_info.model_link = model_link
        new_download_info.untar_path = result_path
    if new_download_info:
        save_model_download_info(new_download_info)
    return result_path


class TestTFPerformanceModel(TestPerformanceModel):
    def load_model(self, model_name, model_link):
        if not os.path.exists(wget_cache_dir):
            os.mkdir(wget_cache_dir)
        return download_model(model_name, model_link)

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
            clean_wget_cache()
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
