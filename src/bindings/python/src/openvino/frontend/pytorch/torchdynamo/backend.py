import logging
from functools import partial

import torch
from torch._dynamo.backends.common import aot_autograd, mem_efficient_fusion_kwargs
from torch._dynamo.backends.common import fake_tensor_unsupported
from torch._dynamo.backends.registry import register_backend

from openvino.runtime import Core, Type, PartialShape
from openvino.frontend import FrontEndManager
from openvino.frontend.pytorch.decoder import TorchScriptPythonDecoder
from openvino.frontend.pytorch.torchdynamo.partition import Partitioner
from openvino.frontend.pytorch.torchdynamo.execute import execute 
from torch.fx.experimental.proxy_tensor import make_fx

log = logging.getLogger(__name__)

@register_backend
@fake_tensor_unsupported
def openvino(subgraph, example_inputs):
    return fx_openvino(subgraph, example_inputs)

def ts_openvino(subgraph, example_inputs):
    try:
        model = torch.jit.script(subgraph)
        model.eval()
        fr_model = torch.jit.freeze(model)
        
        core = Core()
        fe_manager = FrontEndManager()
        fe = fe_manager.load_by_framework('pytorch')
        dtype_mapping = {
            torch.float32: Type.f32,
            torch.float16: Type.f16,
            torch.int64: Type.i64,
            torch.int32: Type.i32,
            torch.uint8: Type.u8,
            torch.int8: Type.i8,
            torch.bool: Type.boolean
        }
        decoder = TorchScriptPythonDecoder(model)

        im = fe.load(decoder)
        om = fe.convert(im)

        for idx, input_data in enumerate(example_inputs):
            om.inputs[idx].get_node().set_element_type(dtype_mapping[input_data.dtype])
            om.inputs[idx].get_node().set_partial_shape(PartialShape(list(input_data.shape)))
        om.validate_nodes_and_infer_types()

        compiled_model = core.compile_model(om, 'CPU')

        def _call(*args):
            ov_inputs = [a.detach().cpu().numpy() for a in args]
            try:
                res = compiled_model(ov_inputs)
            except Exception as e:
                return subgraph.model.forward(*args)
            result = [torch.from_numpy(res[out]) for out in compiled_model.outputs]
            return result
        return _call
    except Exception as e:
        return subgraph


def fx_openvino(subgraph, example_inputs):
    try:
        model = make_fx(subgraph)(*example_inputs)
        with torch.no_grad():
            model.eval()
        partitioner = Partitioner()
        compiled_model = partitioner.make_partitions(model)
    
        def _call(*args):
            res = execute(compiled_model, *example_inputs, executor="openvino")
            return res
        return _call
    except Exception as e:
        return subgraph
