# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
import sys

from e2e_tests.common.multiprocessing_utils import multiprocessing_run
from e2e_tests.common.ref_collector.provider import ClassProvider


class ONNXRuntimeRunner(ClassProvider):
    """Base class for inferring ONNX models with ONNX Runtime"""

    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

    __action_name__ = "score_onnx_runtime"

    def __init__(self, config):
        """
        ONNXRuntime Runner initialization
        :param config: dictionary with class configuration parameters:
        required config keys:
            model: path to the model for inference
        """
        self.model = config["model"]
        self.ep = config["onnx_rt_ep"] if isinstance(config["onnx_rt_ep"], list) else [config["onnx_rt_ep"]]
        self.cast_input_data = config.get("cast_input_data", True)
        self.cast_input_data_to_type = config.get("cast_input_data_to_type", "float32")
        self.res = None
        self.inputs = config["inputs"]

    def run_rt(self, input_data):
        """Return ONNX model reference results."""
        import onnxruntime as rt

        log.info("Loading ONNX model from {} ...".format(self.model))
        opts = rt.SessionOptions()
        sess = rt.InferenceSession(self.model, sess_options=opts)
        if self.ep == [None]:
            log.warning("Execution provider is not specified for ONNX Runtime tests. "
                        "Using CPUExecutionProvider by default.")
            self.ep = ["CPUExecutionProvider"]
        if not all([ep in sess.get_providers() for ep in self.ep]):
            raise ValueError(f"{self.ep} execution provider is not known to ONNX Runtime. "
                             f"Available execution providers: {str(sess.get_providers())}")
        sess.set_providers(self.ep)
        providers_set = sess.get_providers()
        log.info("Using {} as an execution provider.".format(str(providers_set)))
        if self.cast_input_data:
            for layer, data in input_data.items():
                input_data[layer] = data.astype(self.cast_input_data_to_type)
        if len(input_data) > 1:
            log.warning("ONNX Runtime runner is not properly tested to work with multi-input topologies. "
                        "Please, contact QA.")
        for layer in sess.get_inputs():
            model_shape_to_compare = tuple([layer.shape[dim] for dim in range(len(layer.shape))
                                            if not ((layer.shape[dim] is None) or (isinstance(layer.shape[dim], str)))])
            data_shape_to_compare = tuple([input_data[layer.name].shape[dim] for dim in range(len(layer.shape))
                                           if not ((layer.shape[dim] is None) or (isinstance(layer.shape[dim], str)))])
            if model_shape_to_compare != data_shape_to_compare:
                    raise ValueError(f"Shapes of input data {list(input_data.values())[0].shape} and "
                                     f"input blob {sess.get_inputs()[0].shape} are not equal for layer {layer.name}")
        output_names = [output.name for output in sess.get_outputs()]
        if len(output_names) > 1:
            log.warning("ONNX Runtime runner is not properly tested to work with multi-output topologies. "
                        "Please, contact QA.")
        log.info("Starting inference with ONNX Runtime ...".format(self.model))
        out = sess.run(output_names, input_data)
        res = {output_names[i]: out[i] for i in range(len(output_names))}
        return res

    def get_refs(self):
        self.res = multiprocessing_run(self.run_rt, [self.inputs], "ONNX Runtime Inference", timeout=200)
        return self.res
