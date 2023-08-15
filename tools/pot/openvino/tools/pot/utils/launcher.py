# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.runtime import Core, PartialShape  # pylint: disable=E0611,E0401

from ..utils.utils import create_tmp_dir
from ..graph.model_utils import save_model
from ..graph.node_utils import convert_to_outputs_name
from ..engines.utils import cast_friendly_names, process_raw_output


class IELauncher:
    """ Inference Engine Launcher for model inference """

    def __init__(self, device='CPU'):
        """ Constructor
         :param device: specify the target device to infer on; CPU, GPU or HETERO: is acceptable
         :param extension: path to the extension library with custom layers
        """
        self._tmp_dir = create_tmp_dir()
        self.device = device
        self.model = None
        self.infer_request = None

        self._ie = Core()
        self._ie.set_property({"ENABLE_MMAP": "NO"})

    def set_model(self, model, output_names=None, md_shapes=None):
        """ Set/reset model to instance of engine class
         :param model: CompressedModel instance for inference
        """
        if model.is_cascade:
            raise Exception('Cascade models are not supported in current launcher')

        # save model in IR
        path = save_model(model, self._tmp_dir.name, 'tmp_model')[0]
        # load IR model
        ir_model = self._load_model(path)

        cast_friendly_names(ir_model.inputs + ir_model.outputs)

        if output_names is not None:
            output_names = [convert_to_outputs_name(output_name) for output_name in output_names]
            ir_model.add_outputs(output_names)

        if md_shapes is not None:
            ng_shapes = {}
            for key, shape in md_shapes.items():
                ng_shapes[key] = PartialShape(shape)
            ir_model.reshape(ng_shapes)

        self.model = self._ie.compile_model(model=ir_model, device_name=self.device)

        self.infer_request = self.model.create_infer_request()

    def infer(self, inputs):
        """ Inference model
         :param inputs: dictionary of inputs {node_name, value}
         :returns dictionary of outputs {node_name, value}
        """
        outputs = self.infer_request.infer(inputs=inputs)
        return process_raw_output(outputs)

    def _load_model(self, path):
        """ Loads IT model from disk
        :param path: dictionary:
        'model': path to xml
        'weights': path to bin
        :return IE model instance
        """
        return self._ie.read_model(model=path['model'], weights=path['weights'])
