import logging
import os
import torch
from torch._dynamo.backends.common import fake_tensor_unsupported
from torch._dynamo.backends.registry import register_backend
from torch._inductor.compile_fx import compile_fx

from openvino.runtime import Core, Type, PartialShape
from openvino.frontend import FrontEndManager
from openvino.frontend.pytorch.decoder import TorchScriptPythonDecoder

log = logging.getLogger(__name__)


@register_backend
@fake_tensor_unsupported
def openvino(subgraph, example_inputs):
    return ts_openvino(subgraph, example_inputs)


def ts_openvino(subgraph, example_inputs):
    try:
        model = torch.jit.script(subgraph)
        model.eval()
        fr_model = torch.jit.freeze(model)

        core = Core()
        fe_manager = FrontEndManager()
        fe = fe_manager.load_by_framework('pytorch')
        dtype_mapping = {
            torch.float64: Type.f64,
            torch.float32: Type.f32,
            torch.float16: Type.f16,
            torch.int64: Type.i64,
            torch.int32: Type.i32,
            torch.uint8: Type.u8,
            torch.int8: Type.i8,
            torch.bool: Type.boolean
        }
        decoder = TorchScriptPythonDecoder(fr_model)

        im = fe.load(decoder)
        om = fe.convert(im)

        for idx, input_data in enumerate(example_inputs):
            om.inputs[idx].get_node().set_element_type(dtype_mapping[input_data.dtype])
            om.inputs[idx].get_node().set_partial_shape(PartialShape(list(input_data.shape)))
        om.validate_nodes_and_infer_types()

        device = 'CPU'
        if (os.getenv("OPENVINO_DEVICE") is not None):
            device = os.getenv("OPENVINO_DEVICE")
            assert device in core.available_devices, "Specified device " + device + " is not in the list of OpenVINO Available Devices"

        compiled_model = core.compile_model(om, device)

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
        print(f"Exception is {str(e)}")
        return compile_fx(subgraph, example_inputs)