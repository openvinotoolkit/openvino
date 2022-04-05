# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys

from common.utils.multiprocessing_utils import multiprocessing_run
from openvino.inference_engine import IECore, get_version as ie_get_version
from openvino.runtime import Core, get_version as ie2_get_version


def shell(cmd, env=None, cwd=None):
    if sys.platform.startswith('linux') or sys.platform == 'darwin':
        cmd = ['/bin/bash', '-c', "".join(cmd)]
    else:
        cmd = "".join(cmd)
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (stdout, stderr) = p.communicate()
    return p.returncode, stdout, stderr


class BaseInfer:
    def __init__(self, name):
        self.name = name
        self.res = None

    def fw_infer(self, input_data):
        raise RuntimeError("This is base class, please implement infer function for the specific framework")

    def get_inputs_info(self, precision) -> dict:
        raise RuntimeError("This is base class, please implement get_inputs_info function for the specific framework")

    def infer(self, input_data, infer_timeout=10):
        self.res = multiprocessing_run(self.fw_infer, [input_data], self.name, infer_timeout)
        return self.res


class IEInfer(BaseInfer):
    def __init__(self, model, weights, device):
        super().__init__('Inference Engine')
        self.device = device
        self.model = model
        self.weights = weights

    def fw_infer(self, input_data):

        print("Inference Engine version: {}".format(ie_get_version()))
        print("Creating IE Core Engine...")
        ie = IECore()
        print("Reading network files")
        net = ie.read_network(self.model, self.weights)
        print("Loading network")
        exec_net = ie.load_network(net, self.device)
        print("Starting inference")
        result = exec_net.infer(input_data)

        if "exec_net" in locals():
            del exec_net
        if "ie" in locals():
            del ie

        return result

    def get_inputs_info(self, precision) -> dict:
        core = IECore()
        net = core.read_network(self.model, self.weights)
        inputs_info = {}
        for item in net.input_info.items():
            inputs_info[item[0]] = item[1].tensor_desc.dims

        return inputs_info


class InferAPI20(BaseInfer):
    def __init__(self, model, weights, device):
        super().__init__('Inference Engine')
        self.device = device
        self.model = model
        self.weights = weights

    def fw_infer(self, input_data):
        print("Inference Engine version: {}".format(ie2_get_version()))
        print("Creating IE Core Engine...")
        ie = Core()
        print("Reading network files")
        net = ie.read_model(self.model, self.weights)
        print("Loading network")
        exec_net = ie.compile_model(net, self.device)
        print("Starting inference")
        request = exec_net.create_infer_request()
        request_result = request.infer(input_data)

        result = {}
        for out_obj, out_tensor in request_result.items():
            # all input and output tensors have to be named
            assert out_obj.names, "Output tensor {} has no names".format(out_obj)

            tensor_name = out_obj.get_any_name().split(':')[0]
            result[tensor_name] = out_tensor

        if "exec_net" in locals():
            del exec_net
        if "ie" in locals():
            del ie

        return result

    def get_inputs_info(self, precision) -> dict:
        core = Core()
        net = core.read_model(self.model, self.weights)
        inputs_info = {}
        for item in net.inputs:
            inputs_info[item.get_any_name()] = list(item.shape)

        return inputs_info
