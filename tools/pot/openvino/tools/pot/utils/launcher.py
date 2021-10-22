# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.inference_engine import IECore, IENetwork  # pylint: disable=E0611

from ..utils.utils import create_tmp_dir
from ..graph.model_utils import save_model


class IELauncher:
    """ Inference Engine Launcher for model inference """

    def __init__(self, device='CPU', extension=None):
        """ Constructor
         :param device: specify the target device to infer on; CPU, GPU, FPGA, HDDL, MYRIAD or HETERO: is acceptable
         :param extension: path to the extension library with custom layers
        """
        self._tmp_dir = create_tmp_dir()
        self.device = device
        self.model = None

        self._ie = IECore()
        if extension is not None:
            self._ie.add_extension(extension, device)

    def set_model(self, model, output_names=None, md_shapes=None):
        """ Set/reset model to instance of engine class
         :param model: NXModel instance for inference
        """
        if model.is_cascade:
            raise Exception('Cascade models are not supported in current launcher')

        # save model in IR
        path = save_model(model, self._tmp_dir.name, 'tmp_model')[0]
        # load IR model
        ir_model = self._load_model(path)

        if output_names is not None:
            ir_model.add_outputs(output_names)

        if md_shapes is not None:
            ir_model.reshape(md_shapes)

        self.model = self._ie.load_network(network=ir_model, device_name=self.device)

    def infer(self, inputs):
        """ Inference model
         :param inputs: dictionary of inputs {node_name, value}
         :returns dictionary of outputs {node_name, value}
        """
        return self.model.infer(inputs=inputs)

    def _load_model(self, path):
        """ Loads IT model from disk
        :param path: dictionary:
        'model': path to xml
        'weights': path to bin
        :return IE model instance
        """
        if 'read_network' in IECore.__dict__:
            return self._ie.read_network(model=path['model'], weights=path['weights'])
        return IENetwork(model=path['model'], weights=path['weights'])
