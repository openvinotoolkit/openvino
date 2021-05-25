import os

import mxnet as mx
from common.layer_test_class import CommonLayerTest


class CommonMXNetLayerTest(CommonLayerTest):
    @staticmethod
    def save_to_mxnet(framework_model, path_to_saved_mxnet_model):
        symbol_path = os.path.join(path_to_saved_mxnet_model, 'model-symbol.json')
        params_path = os.path.join(path_to_saved_mxnet_model, 'model-0000.params')
        framework_model['symbol'].save(symbol_path)
        mx.nd.save(params_path, framework_model['params'])
        return params_path

    def produce_model_path(self, framework_model, save_path):
        return CommonMXNetLayerTest.save_to_mxnet(framework_model, save_path)

    def get_framework_results(self, inputs_dict, model_path):
        ctx = mx.cpu()

        # - Symbol will be loaded from ``prefix-symbol.json``.
        # - Parameters will be loaded from ``prefix-epoch.params``.
        parent_path = os.path.dirname(os.path.normpath(model_path))
        prefix = os.path.join(parent_path, 'model')
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, 0)

        mx_data = {}
        for name, data in inputs_dict.items():
            mx_data.update({name: mx.nd.array(data)})
        mx_data_names = list(mx_data.keys())
        batch_size = next(iter(mx_data.values())).shape[0]

        data_iter = mx.io.NDArrayIter(
            data=mx_data, label=None, batch_size=batch_size)
        data_batch = mx.io.DataBatch(data=data_iter.data_list)

        mod = mx.mod.Module(symbol=sym, data_names=mx_data_names, context=ctx, label_names=None)
        mod.bind(for_training=False, data_shapes=data_iter.provide_data)
        mod.set_params(arg_params, aux_params, allow_missing=True, allow_extra=True)
        mod.forward(data_batch)
        result = dict()
        for layer, out in zip(mod.output_names, mod.get_outputs()):
            result[layer.replace('_output', '')] = out.asnumpy()
        return result
