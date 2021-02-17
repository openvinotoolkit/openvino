import argparse
import logging as log
import sys

log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
import os

import numpy as np
from openvino.inference_engine import IENetwork


def python_api_infer(ir_path, device):
    """
     Function to perform IE inference using python API "in place"
    :param device: Device name for inference
    :return: Dict containing out blob name and out data
    """

    from openvino.inference_engine import IECore
    bin_path = os.path.splitext(ir_path)[0] + '.bin'
    net = IENetwork(model=ir_path, weights=bin_path)
    feed_dict = {}
    for layer, name in net.inputs.items():
        feed_dict.update({layer: np.ones(shape=name.shape)})
    ie = IECore()
    exec_net = ie.load_network(net, device)
    res = exec_net.infer(inputs=feed_dict)

    del net
    # It's important to delete executable network first to avoid double free in plugin offloading.
    # Issue relates ony for hetero and Myriad plugins
    del exec_net
    del ie
    return res


def cli_parser():
    parser = argparse.ArgumentParser(description='Python_api reproducer')
    parser.add_argument('-m', dest='ir_path', required=True, help='Path to XML file of IR')
    parser.add_argument('-d', dest='device', help='Target device to infer on', default="CPU")
    parser.add_argument('-r', dest='out_path', default=None,
                        help='Dumps results to the output folder')
    args = parser.parse_args()
    ir_path = args.ir_path
    device = args.device
    out_path = args.out_path
    return ir_path, device, out_path


if __name__ == "__main__":
    ir_path, device, out_path = cli_parser()
    bin_path = os.path.splitext(ir_path)[0] + '.bin'
    results = python_api_infer(ir_path=ir_path, device=device)
    np.savez(out_path, **results)
    log.info("Path for inference results: {}".format(out_path))
    log.info("Inference results:")
    log.info(results)
    log.info("SUCCESS!")