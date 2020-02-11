import argparse
import logging as log
import sys

log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
import os

import numpy as np
from openvino.inference_engine import IENetwork


def python_api_infer(net, feed_dict, device, lib, api, nireq, outputs_to_add: list = None):
    """
     Function to perform IE inference using python API "in place"
    :param net: IENetwork instance
    :param feed_dict: Dict which contains mapping between input blob and input data
    :param device: Device name for inference
    :param lib: Absolute path to custom kernel lib
    :param outputs_to_add: Layer names list to take output from
    :param api: Defines use synchronous infer or asynchronous
    :param nireq: Number of infer requests to create for asynchronous infer
    :return: Dict containing out blob name and out data
    """

    from openvino.inference_engine import IECore
    ie = IECore()

    if outputs_to_add:
        net.add_outputs(outputs_to_add)

    exec_net = ie.load_network(net, device, num_requests=nireq)

    if api == "async":
        res = []
        for i in range(nireq):
            reqest_handler = exec_net.start_async(request_id=i, inputs=feed_dict)
            reqest_handler.wait()
            res.append(reqest_handler.outputs)
    else:
        res = exec_net.infer(inputs=feed_dict)
    del net
    # It's important to delete executable network first to avoid double free in plugin offloading.
    # Issue relates ony for hetero and Myriad plugins
    del exec_net
    del ie
    return res


def cli_parser():
    parser = argparse.ArgumentParser(description='Python_api reproducer')
    parser.add_argument('-i', dest='feed_dict', required=True, help='Path to input data in .npz format')
    parser.add_argument('-m', dest='ir_path', required=True, help='Path to XML file of IR')
    parser.add_argument('-d', dest='device', required=True, help='Target device to infer on')
    parser.add_argument('-api', dest='api', default='sync', help='')
    parser.add_argument('-nireq', dest='nireq', default=1, help='')
    parser.add_argument('-r', dest='out_path', default=None,
                        help='Dumps results to the output folder')
    parser.add_argument('--out_layers', dest='out_layers', default=[],
                        help='Names of layers to dump inference results. Example: "input,conv3d"')
    parser.add_argument('--dump_all_layers', dest='dump_all_layers', default=False, action="store_true",
                        help='Bool value to dump inference results from all layers')

    args = parser.parse_args()
    feed_dict = args.feed_dict
    ir_path = args.ir_path
    device = args.device
    lib = args.lib
    api = args.api
    nireq = int(args.nireq)
    out_path = args.out_path
    if out_path and not os.path.exists(out_path):
        os.makedirs(out_path)
    out_layers = args.out_layers.split(",") if args.out_layers else args.out_layers
    dump_all_layers = args.dump_all_layers
    if out_layers and dump_all_layers:
        raise AttributeError('CMD arguments "out_layers" and "dump_all_layers" were specified together. '
                             'Please, specify only one argument')
    return feed_dict, ir_path, device, lib, api, nireq, out_path, out_layers, dump_all_layers


if __name__ == "__main__":
    feed_dict, ir_path, device, lib, api, nireq, out_path, out_layers, dump_all_layers = cli_parser()

    bin_path = os.path.splitext(ir_path)[0] + '.bin'
    feed_dict = dict(np.load(feed_dict))
    network = IENetwork(model=ir_path, weights=bin_path)
    if dump_all_layers:
        out_layers = list(network.layers.keys())
    results = python_api_infer(net=network, feed_dict=feed_dict, device=device, lib=lib, api=api, nireq=nireq,
                               outputs_to_add=out_layers)
    if out_path:
        if api == "async":
            for i, result in enumerate(results):
                dump_path = os.path.join(out_path, "dump_req{}.npz".format(str(i)))
                np.savez(dump_path, **result)
                log.info("Path for inference results for {} request: {}".format(str(i), dump_path))
        else:
            dump_path = os.path.join(out_path, "dump.npz")
            np.savez(os.path.join(out_path, "dump.npz"), **results)
            log.info("Path for inference results: {}".format(dump_path))
    else:
        log.info("Inference results won't be saved in the file. "
                 "To do it need to specify '-r' option.")
    log.info("Inference results:")
    log.info(results)
    log.info("SUCCESS!")
