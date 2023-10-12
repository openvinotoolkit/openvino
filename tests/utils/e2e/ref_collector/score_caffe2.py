import logging as log
import sys

from utils.multiprocessing_utils import multiprocessing_run
from .provider import ClassProvider


class Caffe2Runner(ClassProvider):
    """Base class for infering ONNX models with Caffe2"""

    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

    __action_name__ = "score_caffe2"

    def __init__(self, config):
        """
        Caffe2Runner initialization
        :param config: dictionary with class configuration parameters:
        required config keys:
            model: path to the model for inference
        """
        self.model = config["model"]
        self.cast_input_data_to_type = config.get("cast_input_data_to_type", "float32")
        self.timeout = config.get("timeout", None)
        self.res = None

    def _get_refs(self, input_data):
        import caffe2.python.onnx.backend
        import onnx

        result = {}
        log.info("Running inference with caffe2 ...")
        log.info("Loading ONNX model from {} ...".format(self.model))
        model = onnx.load(self.model)
        for layer, data in input_data.items():
            input_data[layer] = data.astype(self.cast_input_data_to_type)

        log.info("Starting inference with caffe2 ...")
        out_res = caffe2.python.onnx.backend.run_model(model, input_data)
        self.res = {}
        for out in out_res._fields:
            result[out] = getattr(out_res, out)

        if "model" in locals():
            del model

        return result

    def get_refs(self, input_data):
        self.res = multiprocessing_run(self._get_refs, [input_data], "Caffe2 Inference", self.timeout)
        return self.res
