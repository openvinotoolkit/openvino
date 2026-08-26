# torch_export.py вЂ” PyTorch -> standalone Vulkan core exporter.
#
# Usage:
#   python torch_export.py --model my_module.py:MyModule --weights out.weights.safetensors --graph out.graph.vktorch
#   (or import and call export_with_reference(module, inputs, graph, weights, expected))
#
# Produces:
#   <graph>.vktorch        вЂ” line-based graph (OP/CONST/OUT records)
#   <weights>.safetensors  вЂ” every parameter/buffer, key == tensor id
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


def _inlined_graph(traced):
    # torch <=2.12: method; torch >=2.13: property.
    ig = traced.inlined_graph
    return ig() if callable(ig) else ig


def export(module, example_inputs, graph_path, weights_path):
    module = module.eval()
    traced = torch.jit.trace(module, example_inputs)
    g = _inlined_graph(traced)

    lines = ["# vktorch v1"]
    weights = {}
    value_id = {}
    counter = [0]

    def vid(v):
        if v not in value_id:
            counter[0] += 1
            value_id[v] = f"v{counter[0]}"
        return value_id[v]

    # Module parameters surface through prim::GetAttr chains: GetAttr(self,
    # "fc1") -> GetAttr(that, "weight") == state_dict key "fc1.weight".
    sd = module.state_dict()
    attr_path = {}  # jit value -> dotted attribute path
    for node in g.nodes():
        if node.kind() != "prim::GetAttr":
            continue
        base = attr_path.get(node.input(), "")
        name = node.s("name")
        attr_path[node.output()] = f"{base}.{name}" if base else name

    # Trace inputs that are NOT parameter-carriers = the example_inputs.
    for v in g.inputs():
        if v in attr_path:
            continue
        iid = f"in_{len(value_id)}"
        value_id[v] = iid
        try:
            shape = [int(d) for d in v.type().sizes()]
        except Exception:
            shape = []
        dims = "x".join(map(str, shape)) or "scalar"
        lines.append(f"OP {iid} parameter {dims} -")

    def ensure_param(v):
        """Emits a CONST record for a parameter value on first use."""
        if v in value_id:
            return value_id[v]
        key = attr_path.get(v)
        if key is None or key not in sd:
            return None
        skey = "p_" + key.replace(".", "_")
        weights[skey] = sd[key]
        t = sd[key]
        dims = "x".join(map(str, t.shape)) or "scalar"
        lines.append(f"CONST {skey} {dims} {skey}")
        value_id[v] = skey
        return skey

    for node in g.nodes():
        kind = node.kind()
        if kind in ("prim::Param", "prim::Return"):
            continue
        if kind.startswith("prim::GetAttr") or kind.startswith("prim::Constant"):
            continue
        if kind.startswith("prim::"):
            raise RuntimeError(f"unsupported control-flow node: {kind} (flatten the module)")
        canonical = _KIND_MAP.get(kind)
        if canonical is None:
            raise RuntimeError(f"unsupported torch op: {kind}")
        outs = list(node.outputs())
        out = outs[0]
        oid = vid(out)
        ins = []
        for i in node.inputs():
            k = ensure_param(i)
            ins.append(k if k is not None else vid(i))
        try:
            shape = [int(d) for d in out.type().sizes()]
        except Exception:
            shape = []
        lines.append(f"OP {oid} {canonical} {'x'.join(map(str, shape)) or 'scalar'} {','.join(ins) or '-'}")

    for r in g.return_node().inputs():
        lines.append(f"OUT {vid(r)}")

    # Drop parameter nodes that no OP consumes (e.g. the module self-value).
    referenced = set()
    for ln in lines:
        if ln.startswith("OP ") or ln.startswith("CONST "):
            parts = ln.split()
            referenced.update(parts[4].split(",")) if len(parts) > 4 and parts[4] != "-" else None
    kept = []
    for ln in lines:
        if ln.startswith("OP ") and " parameter " in ln:
            if ln.split()[1] not in referenced:
                continue
        kept.append(ln)
    lines = kept

    with open(graph_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    _write_safetensors(weights_path, weights)
    print(f"exported: {graph_path}, {weights_path} ({len(weights)} tensors)")


def export_with_reference(module, example_inputs, graph_path, weights_path, expected_path):
    """export() + writes the eager-mode output as text (one float per line)
    so the C++ side can verify numerically."""
    with torch.no_grad():
        ref = module(*example_inputs).detach().to("cpu").contiguous().float().numpy().reshape(-1)
    with open(expected_path, "w") as f:
        f.write("\n".join(f"{v:.6f}" for v in ref) + "\n")
    print(f"reference: {expected_path} ({ref.size} values)")
    export(module, example_inputs, graph_path, weights_path)


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
    export_with_reference(model, (dummy,), args.graph, args.weights,
                          args.graph + ".expected.txt")

