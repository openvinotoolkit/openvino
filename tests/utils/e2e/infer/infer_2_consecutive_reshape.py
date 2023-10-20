"""Inference engine runners."""
import logging as log
# pylint:disable=import-error
import sys

from openvino.inference_engine import IECore, get_version as ie_get_version
from e2e_oss.utils.path_utils import resolve_file_path

# import local modules:
from .common_inference import Infer
from .network_modifiers.network_modifiers import Reshape

log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)


class Infer2ConsecutiveReshape(Infer):
    """Inference engine runner with 2 consecutive reshape, load to plugin and inference."""
    __action_name__ = "ie_2_consecutive_reshape"

    def __init__(self, config):
        self.new_shapes = config["new_shapes"]
        self.original_shapes = config["original_shapes"]
        super().__init__(config=config)

    def _infer(self, input_data):
        log.info("Inference Engine version: {}".format(ie_get_version()))
        log.info("Creating IE Core Engine...")
        ie = IECore()
        self._configure_plugin(ie)
        log.info("Loading network files")
        net = ie.read_network(model=str(resolve_file_path(self.xml)), weights=str(resolve_file_path(self.bin)))
        self.network_modifiers.execute(network=net)
        Reshape(config={"shapes": self.new_shapes}).apply(network=net)
        exec_net = ie.load_network(net, self.device)
        log.info("Starting inference")
        exec_net.infer(input_data)

        Reshape(config={"shapes": self.original_shapes}).apply(network=net)
        exec_net = ie.load_network(net, self.device)
        log.info("Starting inference")
        result = exec_net.infer(input_data)

        if "exec_net" in locals():
            del exec_net
        if "ie" in locals():
            del ie

        return result, -1, -1
