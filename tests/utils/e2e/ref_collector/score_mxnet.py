import logging as log
import os
import sys

from e2e_oss.utils.path_utils import resolve_file_path
from .provider import ClassProvider

os.environ['GLOG_minloglevel'] = '3'


class ScoreMxnet(ClassProvider):
    """Reference collector for MXNet models."""
    __action_name__ = "score_mxnet"
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

    def __init__(self, config):
        self.symbol = resolve_file_path(config["symbol"], as_str=True)
        self.params = resolve_file_path(config["params"], as_str=True)
        self.res = {}

    def get_refs(self, input_data):
        """Return MXNet model reference results."""
        log.info("Running inference with mxnet ...")

        import mxnet
        ctx = mxnet.cpu()
        # take full path + name without "-NNNN" part
        model_path = os.path.splitext(self.params)[0][:-5]
        sym, arg_params, aux_params = mxnet.model.load_checkpoint(model_path, 0)

        mx_data = {}
        for name, data in input_data.items():
            mx_data.update({name: mxnet.nd.array(data)})
        mx_data_names = list(mx_data.keys())
        batch_size = next(iter(mx_data.values())).shape[0]

        data_iter = mxnet.io.NDArrayIter(
            data=mx_data, label=None, batch_size=batch_size)
        data_batch = mxnet.io.DataBatch(data=data_iter.data_list)

        mod = mxnet.mod.Module(symbol=sym, data_names=mx_data_names, context=ctx)
        mod.bind(for_training=False, data_shapes=data_iter.provide_data)
        mod.set_params(arg_params, aux_params, allow_missing=True, allow_extra=True)
        mod.forward(data_batch)

        for layer, out in zip(mod.output_names, mod.get_outputs()):
            self.res[layer.replace('_output', '')] = out.asnumpy()

        log.info("MXNet reference collected successfully\n")

        return self.res


def load_params(params: dict):
    """Load params from a file
    """
    arg_params = {}
    aux_params = {}
    for k, v in params.items():
        arg_params[k] = v
    return arg_params, aux_params


class ScoreMxnetV2(ScoreMxnet):
    """Reference collector for MXNet models.
    Differs from the basic version by processing model parameters via load_params function.
    """
    __action_name__ = "score_mxnet_v2"

    def get_refs(self, input_data):
        """Return MXNet model reference results."""
        log.info("Running inference with mxnet ...")

        import mxnet
        ctx = mxnet.cpu()

        params = mxnet.nd.load(self.params)
        symbol = mxnet.sym.load(self.symbol)
        arg_params, aux_params = load_params(params)

        mx_data = {}
        for name, data in input_data.items():
            mx_data.update({name: mxnet.nd.array(data)})
        mx_data_names = list(mx_data.keys())
        batch_size = next(iter(mx_data.values())).shape[0]

        data_iter = mxnet.io.NDArrayIter(
            data=mx_data, label=None, batch_size=batch_size)
        data_batch = mxnet.io.DataBatch(data=data_iter.data_list)

        mod = mxnet.mod.Module(symbol=symbol, data_names=mx_data_names, context=ctx)
        mod.bind(for_training=False, data_shapes=data_iter.provide_data)
        mod.set_params(arg_params, aux_params, allow_missing=True, allow_extra=True)
        mod.forward(data_batch)

        for layer, out in zip(mod.output_names, mod.get_outputs()):
            self.res[layer.replace('_output', '')] = out.asnumpy()

        log.info("MXNet reference collected successfully\n")

        return self.res
