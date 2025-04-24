# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
import os
import sys

from e2e_tests.common.multiprocessing_utils import multiprocessing_run
from utils.path_utils import resolve_dir_path
from e2e_tests.common.ref_collector.score_onnx_runtime import ONNXRuntimeRunner
from e2e_tests.common.ref_collector.provider import ClassProvider


class PyTorchToOnnxRunner:
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    """
    Base class for converting PyTorch models to ONNX format and infering with ONNX Runtime
    
    PyTorch net objects doesn't fully support pickling https://github.com/pytorch/pytorch/issues/49260 and running with
    caffe2 after dumping to ONNX https://github.com/pytorch/pytorch/issues/49752
    
    To get and infer PyTorch pretrained or torchvision model with multiprocessing to avoid potential crashing of main
    process, net is converted to ONNX format and inferred with OnnxInfer class. 
    """
    def __init__(self, config):
        """
        PyTorchToOnnxRunner initialization
        :param config: dictionary with class configuration parameters:
        required config keys:
            model_name: name of the model which will be used in _get_model() function
            torch_model_zoo: path to the folder with pytorch pretrained\torchvision model's weights files
        optional config keys:
            onnx_dump_path: path to the file or folder where to dump onnx model's representation.
                            if onnx_dump_path specified as a directory, target dump file name will be constructed from
                            the path specified in config and model_name attribute + .onnx extension
        """
        self.inputs = config["inputs"]
        self.model_name = config["model_name"]
        self.torch_model_zoo_path = config["torch_model_zoo_path"]
        os.environ['TORCH_HOME'] = os.path.join(self.torch_model_zoo_path, self.model_name)
        self.get_model_args = config.get("get_model_args", {})
        self.onnx_dump_path = config.get("onnx_dump_path")
        self.onnx_model_path = multiprocessing_run(self._get_model, [], "Pytorch Get Model")

    def _get_model(self):
        """
        `_get_model` function have to be implemented in inherited classes
        depending on PyTorch models source (pretrained or torchvision)
        """
        raise NotImplementedError("{}\nDo not use {} class directly!".format(self._get_model.__doc__,
                                                                             self.__class__.__name__))

    def get_refs(self):
        """
        Run inference with Onnx runner
        Note: input_data for the function have to be represented as a dictionary to keep uniform interface
        across all framework's scoring classes. But PyTorch models doesn't have named inputs so the keys
        of the dictionary can have arbitrary value. The first numpy ndarray from input_data.values() will be used
        as input data
        :param input_data: dict with input data for the model
        :return: numpy ndarray with inference results
        """
        runner = ONNXRuntimeRunner({"model": self.onnx_model_path,
                                    "onnx_rt_ep": "CPUExecutionProvider"})
        res = runner.get_refs(self.inputs)
        self.res = {"output": next(iter(res.values()))}
        return self.res

    def _pytorch_to_onnx(self, net):
        """
        Convert and save PyTorch model in ONNX format
        :return: saved ONNX model path
        """
        log.info("Dumping torch model to ONNX ...")
        import torch
        dump_dir = os.path.dirname(self.onnx_dump_path)

        if not os.path.exists(dump_dir):
            log.warning("Target dump directory {} doesn't exist! Let's try to create it ...".format(dump_dir))
            os.makedirs(dump_dir, mode=0o755, exist_ok=True)
            log.warning("{} directory created!".format(dump_dir))
            dump_dir = resolve_dir_path(os.path.dirname(dump_dir), as_str=True)

        # If user defined onnx_dump_path attribute as a folder, target file name will be constructed
        # using self.model_name and joined to the specified path
        if os.path.isdir(self.onnx_dump_path):
            log.warning("Specified ONNX dump path is a directory...")
            model_path = os.path.join(dump_dir, self.model_name + ".onnx")
            log.warning("Target model will be saved with specified model name as {}".format(self.onnx_dump_path))
        else:
            model_path = self.onnx_dump_path
        if os.path.exists(model_path):
            log.warning(
                "Specified ONNX model {} already exist and will not be dumped again".format(model_path))
        else:
            dummy_input = torch.autograd.Variable(torch.randn([1, ] + list(net.input_size)), requires_grad=False)
            torch.onnx.export(net, dummy_input, model_path, export_params=True)
        return model_path


class PytorchPretrainedToONNXRunner(ClassProvider, PyTorchToOnnxRunner):
    """
    PyTorch Pretrained models inference class
    """
    __action_name__ = "score_pytorch_pretrained_with_onnx"

    def _get_model(self):
        """
        Get PyTorch model implemented in `pretrained` module
        :return: path to dumped onnx object
        """
        log.info("Getting PyTorch pretrained model ...")
        import pretrainedmodels
        net = getattr(pretrainedmodels.models, self.model_name)(**self.get_model_args)
        return self._pytorch_to_onnx(net)


class PytorchTorchvisionToONNXRunner(ClassProvider, PyTorchToOnnxRunner):
    """
    PyTorch Torchvision models inference class
    """
    __action_name__ = "score_pytorch_torchvision_with_onnx"

    def __init__(self, config):
        """
        PytorchTorchvisionRunner initialization
        :param config: dictionary with class configuration parameters:
        required and optional config keys are the same as in parent PytorchBaseRunner class plus optional key
        `input_size` used to dump model to onnx format ([3,224,224] by default since most of the torchvision models have
        such input size)
        """
        self.input_size = config.get("input_size", [3, 224, 224])
        PyTorchToOnnxRunner.__init__(self, config=config)

    def _get_model(self):
        """
        Get PyTorch model implemented in `torchvision` module
        :return: path to dumped onnx object
        """
        log.info("Getting PyTorch torchvision model ...")
        import torchvision
        net = getattr(torchvision.models, self.model_name)(pretrained=True, **self.get_model_args)
        """
        Torchvision models doesn't have information about input size like in pretrained models.
        `input_size` attribute will be set manually to keep` _pytorch_to_onnx` function implementation
        uniform for pretrained and torchvision pytorch models
        """
        setattr(net, "input_size", self.input_size)
        return self._pytorch_to_onnx(net)
