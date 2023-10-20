import logging as log
import os
import sys

from e2e_oss.common_utils.multiprocessing_utils import multiprocessing_run
from e2e_oss.utils.path_utils import resolve_file_path
from .provider import ClassProvider

os.environ['GLOG_minloglevel'] = '3'


class ScoreCaffe(ClassProvider):
    """Reference collector for Caffe models."""
    __action_name__ = "score_caffe"
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

    def __init__(self, config):
        self.prototxt = resolve_file_path(config["proto"], as_str=True)
        self.model = resolve_file_path(config["model"], as_str=True)
        self.timeout = config.get("timeout", None)
        self.res = None

    def _get_refs(self, input_data):
        """Return Caffe model reference results."""
        import caffe

        result = {}
        log.info("Running inference with Caffe ...")
        caffe.set_mode_cpu()
        log.info("Loading PROTOTXT file from {} ...".format(self.prototxt))
        log.info("Loading MODEL from {} ...".format(self.model))
        net = caffe.Net(self.prototxt, self.model, caffe.TEST)
        for input, data in input_data.items():
            if data.shape[1:] != net.blobs[input].data.shape[1:]:
                raise ValueError(
                    "Shapes of input data {} and input blob {} are not equal".format(data.shape,
                                                                                     net.blobs[input].data.shape))
            elif data.shape[0] != net.blobs[input].data.shape[0]:
                log.warning(
                    "Network will be reshaped from shape {} to {}".format(net.blobs[input].data.shape, data.shape))
                net.blobs[input].reshape(*data.shape)
            for i in range(data.shape[0]):
                net.blobs[input].data[i] = data[i]

        log.info("Starting inference with caffe ...")

        net.forward()
        for out in net.outputs:
            for layer in net._layer_names:
                if out in net.top_names[layer]:
                    result[layer] = net.blobs[net.top_names[layer][0]].data
        log.info("Caffe reference collected successfully\n")

        if "net" in locals():
            del net

        return result

    def get_refs(self, input_data):
        self.res = multiprocessing_run(self._get_refs, [input_data], "Caffe Inference", self.timeout)
        return self.res
