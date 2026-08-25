# torch_export.py — PyTorch -> standalone Vulkan core exporter.
#
# Usage:
#   python torch_export.py --model my_module.py:MyModule --weights out.weights.safetensors --graph out.graph.vktorch
#   (or import and call export(module, example_inputs, graph_path, weights_path))
#
# Produces:
#   <graph>.vktorch        — line-based graph (OP/CONST/OUT records)
#   <weights>.safetensors  — every parameter/buffer, key == tensor id
#
# The C++ side (src/core/pytorch_reader.hpp) consumes both.

import argparse
import importlib
import json

import torch


def _dims(t):
    return "x".join(str(d) for d in t.shape)


def _write_safetensors(path, tensors):
    # tensors: dict[str, torch.Tensor] (contiguous, cpu)
    header = {}
    offset = 0
    blobs = []
    for name in sorted(tensors):
        t = tensors[name].detach().to("cpu").contiguous()
        if t.dtype != torch.float32:
            t = t.float()
        raw = t.numpy().tobytes()
        header[name] = {"dtype": "F32", "shape": list(t.shape),
                        "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
        blobs.append(raw)
    hj = json.dumps(header, separators=(",", ":")).encode()
    while (8 + len(hj)) % 8:
        hj += b" "
    with open(path, "wb") as f:
        f.write(len(hj).to_bytes(8, "little"))
        f.write(hj)
        for b in blobs:
            f.write(b)


_KIND_MAP = {
    "aten::relu": "relu", "aten::relu_": "relu",
    "aten::add": "add", "aten::add_": "add",
    "aten::mul": "mul", "aten::mul_": "mul",
    "aten::sub": "sub", "aten::sub_": "sub",
    "aten::div": "div", "aten::div_": "div",
    "aten::sigmoid": "sigmoid", "aten::tanh": "tanh",
    "aten::leaky_relu": "leaky_relu",
    "aten::gelu": "gelu",
    "aten::softmax": "softmax",
    "aten::transpose": "transpose", "aten::permute": "transpose",
    "aten::view": "reshape", "aten::reshape": "reshape", "aten::flatten": "reshape",
    "aten::cat": "concat",
    "aten::matmul": "matmul", "aten::mm": "matmul", "aten::bmm": "matmul",
    "aten::linear": "linear",
    "aten::conv2d": "conv2d",
    "aten::max_pool2d": "maxpool2d",
    "aten::avg_pool2d": "avgpool2d",
    "aten::argmax": "argmax",
    "aten::mean": "mean", "aten::sum": "sum", "aten::amax": "max",
    "aten::pad": "pad",
    "aten::narrow": "narrow", "aten::slice": "slice",
}


def export(module, example_inputs, graph_path, weights_path):
    module = module.eval()
    traced = torch.jit.trace(module, example_inputs)
    g = traced.inlined_graph()

    lines = ["# vktorch v1"]
    weights = {}
    value_id = {}          # jit value -> bridge id
    counter = [0]

    def vid(v):
        if v not in value_id:
            counter[0] += 1
            value_id[v] = f"v{counter[0]}"
        return value_id[v]

    # parameters/buffers first: every get_attr becomes a CONST fed from safetensors
    for name, tensor in module.state_dict().items():
        key = "p_" + name.replace(".", "_")
        weights[key] = tensor
        value_id[traced_name(g, name)] = key  # helper below resolves graph value

    for node in g.nodes():
        kind = node.kind()
        if kind == "prim::Param":
            for out in node.outputs():
                iid = f"in_{len(value_id)}"
                value_id[out] = iid
                try:
                    shape = [int(d) for d in out.type().sizes()]
                except Exception:
                    shape = []
                dims = "x".join(map(str, shape)) or "scalar"
                lines.append(f"OP {iid} parameter {dims} -")
            continue
        if kind == "prim::GetAttr":
            continue  # handled via state_dict above
        if kind.startswith("prim::"):
            continue  # control flow unsupported -> clear error later on C++ side
        canonical = _KIND_MAP.get(kind)
        if canonical is None:
            raise RuntimeError(f"unsupported torch op: {kind}")
        outs = list(node.outputs())
        out = outs[0]
        oid = vid(out)
        ins = [vid(i) for i in node.inputs()]
        try:
            shape = [int(d) for d in out.type().sizes()]
        except Exception:
            shape = []
        lines.append(f"OP {oid} {canonical} {'x'.join(map(str, shape)) or 'scalar'} {','.join(ins) or '-'}")

    for r in g.return_node().inputs():
        lines.append(f"OUT {vid(r)}")

    with open(graph_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    _write_safetensors(weights_path, weights)
    print(f"exported: {graph_path}, {weights_path} ({len(weights)} tensors)")


def traced_name(g, param_name):
    # TorchScript exposes parameters as values named exactly like state_dict keys.
    for v in g.inputs():
        if v.debugName() == param_name:
            return v
    raise KeyError(param_name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True, help="file.py:ClassName")
    ap.add_argument("--graph", default="model.graph.vktorch")
    ap.add_argument("--weights", default="model.weights.safetensors")
    args = ap.parse_args()
    file_name, cls = args.module.split(":")
    mod = importlib.import_module(file_name[:-3] if file_name.endswith(".py") else file_name)
    model = getattr(mod, cls)()
    model.eval()
    dummy = torch.randn(1, 3, 224, 224)  # adjust per model
    export(model, (dummy,), args.graph, args.weights)
