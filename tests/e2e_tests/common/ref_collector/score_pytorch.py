# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
import os
import sys

from utils.path_utils import resolve_dir_path
from e2e_tests.common.ref_collector.provider import ClassProvider


class PytorchBaseRunner:
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    """
    Base class for inferring models with PyTorch and converting PyTorch models to ONNX format
    """

    def __init__(self, config):
        """
        PyTorchBaseRunner initialization
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
        self.model_name = config.get("model_name")
        self.torch_model_zoo_path = config.get("torch_model_zoo_path", '')
        os.environ['TORCH_HOME'] = self.torch_model_zoo_path
        self.get_model_args = config.get("get_model_args", {})
        self.net = self._get_model()
        self.onnx_dump_path = config.get("onnx_dump_path")
        if config.get('convert_to_onnx'):
            self._pytorch_to_onnx()

    def _get_model(self):
        """
        `_get_model` function have to be implemented in inherited classes
        depending on PyTorch models source (pretrained or torchvision)
        """
        raise NotImplementedError("{}\nDo not use {} class directly!".format(self._get_model.__doc__,
                                                                             self.__class__.__name__))

    def get_refs(self):
        """
        Run inference with PyTorch
        Note: input_data for the function have to be represented as a dictionary to keep uniform interface
        across all framework's scoring classes. But PyTorch models doesn't have named inputs so the keys
        of the dictionary can have arbitrary value. The first numpy ndarray from input_data.values() will be used
        as input data
        :param input_data: dict with input data for the model
        :return: numpy ndarray with inference results
        """
        log.info("Running inference with torch ...")
        import torch
        # PyTorch forward method accepts input data without mapping on input tensor
        # All models from pytorch pretrained have only one input, so we will work with 1st numpy array from input dict
        input_array = next(iter(self.inputs.values()))
        input_variable = torch.autograd.Variable(torch.Tensor(input_array))
        self.net.eval()
        self.res = {"output": self.net(input_variable).detach().numpy()}
        return self.res

    def _pytorch_to_onnx(self):
        """
        Convert and save PyTorch model in ONNX format
        :return:
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
            self.onnx_dump_path = os.path.join(dump_dir, self.model_name + ".onnx")
            log.warning("Target model will be saved with specified model name as {}".format(self.onnx_dump_path))

        if os.path.exists(self.onnx_dump_path):
            log.warning(
                "Specified ONNX model {} already exist and will not be dumped again".format(self.onnx_dump_path))
        else:
            dummy_input = torch.autograd.Variable(torch.randn([1, ] + list(self.net.input_size)), requires_grad=False)
            torch.onnx.export(self.net, dummy_input, self.onnx_dump_path, export_params=True)


class PytorchPretrainedRunner(ClassProvider, PytorchBaseRunner):
    """
    PyTorch Pretrained models inference class
    """
    __action_name__ = "score_pytorch_pretrained"

    def _get_model(self):
        """
        Get PyTorch model implemented in `pretrained` module
        :return: PyTorch Network object
        """
        log.info("Getting PyTorch pretrained model ...")
        import pretrainedmodels
        return getattr(pretrainedmodels.models, self.model_name)(**self.get_model_args)


class PytorchTorchvisionRunner(ClassProvider, PytorchBaseRunner):
    """
    PyTorch Torchvision models inference class
    """
    __action_name__ = "score_pytorch_torchvision"

    def __init__(self, config):
        """
        PytorchTorchvisionRunner initialization
        :param config: dictionary with class configuration parameters:
        required and optional config keys are the same as in parent PytorchBaseRunner class plus optional key
        `input_size` used to dump model to onnx format ([3,224,224] by default since most of the torchvision models have
        such input size)
        """
        self.input_size = config.get("input_size", [3, 224, 224])
        PytorchBaseRunner.__init__(self, config=config)

    def _get_model(self):
        """
        Get PyTorch model implemented in  `torchvision` module
        :return: PyTorch Network object
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
        return net


class PytorchTorchvisionDetectionRunner(ClassProvider, PytorchBaseRunner):
    """
    PyTorch Torchvision models inference class
    """
    __action_name__ = "score_pytorch_torchvision_detection"

    def __init__(self, config):
        """
        PytorchTorchvisionRunner initialization
        :param config: dictionary with class configuration parameters:
        required and optional config keys are the same as in parent PytorchBaseRunner class plus optional key
        `input_size` used to dump model to onnx format ([3,800,800] by default since most of the torchvision detection
        models have such input size)
        """
        self.input_size = config.get("input_size", [3, 800, 800])
        PytorchBaseRunner.__init__(self, config=config)

    def _get_model(self):
        """
        Get PyTorch model implemented in `torchvision.models.detection` module
        :return: PyTorch Network object
        """
        log.info("Getting PyTorch Detection model ...")
        import torchvision
        net = getattr(torchvision.models.detection, self.model_name)(pretrained=True, **self.get_model_args)
        """
        Torchvision Detection models doesn't have information about input size like in pretrained models.
        `input_size` attribute will be set manually to keep` _pytorch_to_onnx` function implementation
        uniform for pytorch models
        """
        setattr(net, "input_size", self.input_size)
        return net

    def get_refs(self):
        """
        Run inference with PyTorch
        Note: input_data for the function have to be represented as a dictionary to keep uniform interface
        across all framework's scoring classes. But PyTorch models doesn't have named inputs so the keys
        of the dictionary can have arbitrary value. The first numpy ndarray from input_data.values() will be used
        as input data
        :param input_data: dict with input data for the model
        :return: numpy ndarray with inference results
        """
        log.info("Running inference with torch ...")
        import torch
        # PyTorch forward method accepts input data without mapping on input tensor
        input_array = next(iter(self.inputs.values()))
        input_variable = torch.autograd.Variable(torch.Tensor(input_array))
        self.net.eval()
        self.res = {"output": self.net(input_variable)[0]}
        return self.res


class PytorchTorchvisionOpticalFlowRunner(ClassProvider, PytorchBaseRunner):
    """
    PyTorch Torchvision models inference class
    """
    __action_name__ = "score_pytorch_torchvision_optical_flow"

    def __init__(self, config):
        """
        PytorchTorchvisionRunner initialization
        :param config: dictionary with class configuration parameters:
        required and optional config keys are the same as in parent PytorchBaseRunner class plus optional key
        `input_size` used to dump model to onnx format ([3,520,960] by default since most of the torchvision optical flow
         models have
        such input size)
        """
        self.input_size = config.get("input_size", [3, 520, 960])
        PytorchBaseRunner.__init__(self, config=config)

    def _get_model(self):
        """
        Get PyTorch model implemented in  `torchvision` module
        :return: PyTorch Network object
        """
        log.info("Getting PyTorch torchvision model ...")
        import torchvision
        net = getattr(torchvision.models.optical_flow, self.model_name)(pretrained=True, **self.get_model_args)
        """
        Torchvision models doesn't have information about input size like in pretrained models.
        `input_size` attribute will be set manually to keep` _pytorch_to_onnx` function implementation
        uniform for pretrained and torchvision pytorch models
        """
        setattr(net, "input_size", self.input_size)
        return net

    def get_refs(self):
        """
        Run inference with PyTorch
        Note: input_data for the function have to be represented as a dictionary to keep uniform interface
        across all framework's scoring classes. But PyTorch models doesn't have named inputs so the keys
        of the dictionary can have arbitrary value.
        :param input_data: dict with input data for the model
        :return: numpy ndarray with inference results
        """
        log.info("Running inference with torch ...")
        import torch
        # PyTorch forward method accepts input data without mapping on input tensor
        input_variable = [torch.autograd.Variable(torch.Tensor(x)) for x in self.inputs.values()]
        assert len(input_variable) == 2, "There should be 2 inputs for optical flow models"
        self.net.eval()
        # We are only interested in the final predicted flows (they are the most accurate ones),
        # so we will just retrieve the last item in the list
        self.res = {"output": self.net(input_variable[0], input_variable[1])}
        self.res["output"] = self.res["output"][-1].detach().numpy()
        return self.res

    def _pytorch_to_onnx(self):
        """
        Convert and save PyTorch model in ONNX format
        :return:
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
            self.onnx_dump_path = os.path.join(dump_dir, self.model_name + ".onnx")
            log.warning("Target model will be saved with specified model name as {}".format(self.onnx_dump_path))

        if os.path.exists(self.onnx_dump_path):
            log.warning(
                "Specified ONNX model {} already exist and will not be dumped again".format(self.onnx_dump_path))
        else:
            dummy_input = torch.autograd.Variable(torch.randn([1, ] + list(self.net.input_size)), requires_grad=False)
            torch.onnx.export(self.net, (dummy_input, dummy_input), self.onnx_dump_path, export_params=True,
                              opset_version=16)


class PytorchTimmRunner(ClassProvider, PytorchBaseRunner):
    """
    PyTorch Torchvision models inference class
    """
    __action_name__ = "score_pytorch_timm"

    def __init__(self, config):
        """
        PytorchTorchvisionRunner initialization
        :param config: dictionary with class configuration parameters:
        required and optional config keys are the same as in parent PytorchBaseRunner class plus optional key
        `input_size` used to dump model to onnx format ([3,224,224] by default since most of the torchvision models have
        such input size)
        """
        self.input_size = config.get("input_size", [3, 224, 224])
        PytorchBaseRunner.__init__(self, config=config)

    def _get_model(self):
        """
        Get PyTorch model implemented in `timm` module
        :return: PyTorch Network object
        """
        log.info("Getting PyTorch Timm model ...")
        import timm
        net = getattr(timm.models, self.model_name)(pretrained=True, **self.get_model_args)
        """
        Timm models doesn't have information about input size like in pretrained models.
        `input_size` attribute will be set manually to keep` _pytorch_to_onnx` function implementation
        uniform for pytorch models
        """
        setattr(net, "input_size", self.input_size)
        return net


class PytorchSavedModelRunner(ClassProvider, PytorchBaseRunner):
    """
    PyTorch saved models inference class
    """
    __action_name__ = "score_pytorch_saved_model"

    def __init__(self, config):
        """
        PytorchTorchvisionRunner initialization
        :param config: dictionary with class configuration parameters:
        required and optional config keys are the same as in parent PytorchBaseRunner class
        """
        self.model_path = config["model-path"]
        self.model_class_path = config.get('model_class_path')
        if self.model_class_path:
            sys.path.insert(0, os.path.abspath(self.model_class_path))
        PytorchBaseRunner.__init__(self, config=config)

    def _get_model(self):
        """
        Load Pytorch model from path
        :return: PyTorch Network object
        """
        log.info("Getting PyTorch saved model ...")
        import torch
        net = torch.load(self.model_path)

        return net

    def get_refs(self):
        """
        Run inference with PyTorch
        :param input_data: input data for the model. Could be list or dict with tensors
        :return: numpy ndarray with inference results
        """
        log.info("Running inference with torch ...")
        self.net.eval()
        if isinstance(self.inputs, dict):
            try:
                self.res = self.net(**self.inputs)
            except Exception as e:
                log.info(f"Tried to infer model with unpacking arguments (self.res = self.net(**input_data)), but got "
                         f"exception: \n{e}")
                self.res = self.net(self.inputs)
        if isinstance(self.inputs, list):
            self.res = self.net(*self.inputs)
        return self.res
