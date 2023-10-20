"""Inference engine runners."""
import logging as log
# pylint:disable=import-error
import sys
import time

from openvino.inference_engine import IECore, get_version as ie_get_version
from e2e_oss.utils.path_utils import resolve_file_path

# import local modules:
from .common_inference import Infer

log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)


class NoInfer(Infer):
    """Inference engine runner without inference."""
    __action_name__ = "ie_no_infer"

    def _infer(self, input_data):
        log.info("Inference Engine version: {}".format(ie_get_version()))
        result = None
        log.info("Creating IE Core Engine...")
        ie = IECore()
        self._configure_plugin(ie)
        log.info("Loading network files")
        net = ie.read_network(model=str(resolve_file_path(self.xml)), weights=str(resolve_file_path(self.bin)))
        self.network_modifiers.execute(network=net)
        # Measure time of loading network to plugin
        t_load_to_pl = time.time()
        exec_net = ie.load_network(net, self.device)
        load_net_to_plug_time = time.time() - t_load_to_pl

        if "exec_net" in locals():
            del exec_net
        if "ie" in locals():
            del ie

        return result, load_net_to_plug_time, -1
