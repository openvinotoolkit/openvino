# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Graph-fingerprint regression tool for the GGUF frontend's native builder.
#
# The fingerprint is a sha256 over the sorted (op_type, output_partial_shape) pairs of every op in
# the converted ov::Model, plus the sorted model input names. It is a cheap, exact regression gate:
# any builder change that alters the produced graph for a given model changes its fingerprint, and
# any change that is meant to be graph-neutral (refactor, robustness, renaming) must leave every
# fingerprint unchanged.
#
# Usage (baseline / manual):
#   PYTHONPATH=<ov>/bin/intel64/Release/python LD_LIBRARY_PATH=<ov>/bin/intel64/Release \
#   python3 graph_fingerprint.py model_a.gguf model_b.gguf ...
#
# It prints "<name> <sighash>" per model. Save the output as the baseline; after a change, re-run
# and diff. The pytest wrapper (test_graph_fingerprint.py) automates this against a JSON baseline
# and is gated on GGUF_FINGERPRINT_MODELS being set (so CI skips when no local models are present).

import hashlib
import sys

import openvino as ov
from openvino.frontend import FrontEndManager


def convert_gguf(model_path: str):
    """Convert a .gguf through the GGUF frontend.

    The frontend is not auto-selectable (see is_hidden_frontend in
    src/frontends/common/src/manager.cpp), so core.read_model(".gguf") does not reach it and it
    has to be requested by name.
    """
    fe = FrontEndManager().load_by_framework("gguf")
    return fe.convert(fe.load(model_path))


def fingerprint(model_path: str) -> dict:
    m = convert_gguf(model_path)
    sig = []
    for op in m.get_ops():
        shape = str(op.get_output_partial_shape(0)) if op.get_output_size() > 0 else ""
        sig.append(op.get_type_name() + "|" + shape)
    inputs = sorted(list(i.get_names())[0] if i.get_names() else "?" for i in m.inputs)
    outputs = sorted(list(o.get_names())[0] if o.get_names() else "?" for o in m.outputs)
    h = hashlib.sha256("\n".join(sorted(sig)).encode()).hexdigest()[:16]
    return {"sighash": h, "num_ops": len(m.get_ops()), "inputs": inputs, "outputs": outputs}


def main(argv):
    if len(argv) < 2:
        print("usage: graph_fingerprint.py <model.gguf> [<model.gguf> ...]", file=sys.stderr)
        return 2
    for path in argv[1:]:
        fp = fingerprint(path)
        print(f"{path}\t{fp['sighash']}\tops={fp['num_ops']}\tinputs={fp['inputs']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
