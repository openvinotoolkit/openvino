# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import math
import os
import subprocess
import sys
import tempfile

import numpy as np
import openvino.runtime.opset12 as ops
import pytest
import torch
from openvino import convert_model, Model, PartialShape, Type
from openvino.frontend import ConversionExtension

from torch_utils import TestTorchConvertModel

# To make tests reproducible we seed the random generator
torch.manual_seed(0)


def custom_op_loop(context):
    map = context.get_input(0)
    points = context.get_input(1)
    kernel_size = context.get_values_from_const_input(2, None, int)
    # kernel_size=2, radius=0.5, pad_left_top=0, pad_right_bottom=1
    # kernel_size=3, radius=1.0, pad_left_top=1, pad_right_bottom=1
    # kernel_size=4, radius=1.5, pad_left_top=1, pad_right_bottom=2
    # kernel_size=5, radius=2.0, pad_left_top=2, pad_right_bottom=2
    radius = (kernel_size - 1.0) / 2.0
    pad_left_top = math.floor(radius)
    pad_right_bottom = math.ceil(radius)

    # pad map: Cx(H+2*radius)x(W+2*radius)
    map_pad = ops.pad(map,
                      np.int32([0, pad_left_top, pad_left_top]),
                      np.int32([0, pad_right_bottom, pad_right_bottom]),
                      "constant",
                      0.)

    # get patches
    points_shape = ops.shape_of(points)
    batch = ops.gather(points_shape, np.int32([0]), 0)
    loop = ops.loop(batch.output(0), ops.constant([True]).output(0))
    points_i = ops.parameter(PartialShape([1, 2]), Type.i64)
    points_i_1d = ops.squeeze(points_i, 0)
    points_i_rev = ops.gather(points_i_1d, np.int32([1, 0]), 0)
    map_body = ops.parameter(PartialShape([-1, -1, -1]), Type.i32)
    points_plus_kenel = ops.add(points_i_rev, np.int64(kernel_size))
    patch_i = ops.slice(
        map_body, points_i_rev, points_plus_kenel, np.int64([1, 1]), np.int64([1, 2]))
    patch_i = ops.unsqueeze(patch_i, 0)
    body = Model([ops.constant([True]), patch_i], [points_i, map_body])
    loop.set_function(body)
    loop.set_special_body_ports([-1, 0])
    loop.set_sliced_input(points_i, points, 0, 1, 1, -1, 0)
    loop.set_invariant_input(map_body, map_pad.output(0))
    res = loop.get_concatenated_slices(patch_i.output(0), 0, 1, 1, -1, 0)
    return [res]


def read_image(path, idx):
    import cv2
    from torchvision.transforms import ToTensor

    img_path = os.path.join(path, f"{idx}.jpg")
    img_ref = cv2.imread(img_path)
    img_ref = cv2.resize(img_ref, (640, 640))
    img_rgb = cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB)
    img_tensor = ToTensor()(img_rgb)
    return img_tensor.unsqueeze_(0)


class TestAlikedConvertModel(TestTorchConvertModel):
    def setup_class(self):
        self.repo_dir = tempfile.TemporaryDirectory()
        os.system(
            f"git clone https://github.com/mvafin/ALIKED.git {self.repo_dir.name}")
        subprocess.check_call(
            ["git", "checkout", "6008af43942925eec7e32006814ef41fbd0858d8"], cwd=self.repo_dir.name)
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "-r", os.path.join(self.repo_dir.name, "requirements.txt")])
        subprocess.check_call(["sh", "build.sh"], cwd=os.path.join(
            self.repo_dir.name, "custom_ops"))

    def load_model(self, model_name, model_link):
        sys.path.append(self.repo_dir.name)
        from nets.aliked import ALIKED

        m = ALIKED(model_name=model_name, device="cpu")
        img_tensor = read_image(os.path.join(
            self.repo_dir.name, "assets", "st_pauls_cathedral"), 1)
        self.example = (img_tensor,)
        img_tensor2 = read_image(os.path.join(
            self.repo_dir.name, "assets", "st_pauls_cathedral"), 2)
        self.inputs = (img_tensor2,)
        return m

    def convert_model(self, model_obj):
        m = convert_model(model_obj,
                          example_input=self.example,
                          extension=ConversionExtension(
                              "custom_ops::get_patches_forward", custom_op_loop)
                          )
        return m

    def infer_fw_model(self, model_obj, inputs):
        fw_outputs = model_obj(*[torch.from_numpy(i) for i in inputs])
        if isinstance(fw_outputs, dict):
            for k in fw_outputs.keys():
                fw_outputs[k] = fw_outputs[k].numpy(force=True)
        elif isinstance(fw_outputs, (list, tuple)):
            fw_outputs = [o.numpy(force=True) for o in fw_outputs]
        else:
            fw_outputs = [fw_outputs.numpy(force=True)]
        return fw_outputs

    def teardown_class(self):
        # remove all downloaded files from cache
        self.repo_dir.cleanup()

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("name", ['aliked-n16rot'])
    def test_convert_model_all_models_default(self, name, ie_device):
        self.run(name, None, ie_device)

    @pytest.mark.nightly
    @pytest.mark.parametrize("name", ['aliked-t16', 'aliked-n16', 'aliked-n32'])
    def test_convert_model_all_models(self, name, ie_device):
        self.run(name, None, ie_device)
