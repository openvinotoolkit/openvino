# -*- coding: utf-8 -*-
# Tests for translate_multi_dot: numerical correctness + order-efficiency.

import numpy as np
import pytest
import torch
import openvino as ov

RTOL = 1e-4
ATOL = 1e-4

# ---------- Helpers ----------

def make_small_tensor(shape, dtype=torch.float32, device="cpu", seed=0):
    """Create a tensor with values in [-10, 10] and at most 1 decimal."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    data = 
        (torch.randint(-100, 101, shape, generator=g, device=device, dtype=torch.int32)
        .to(torch.float32)) / 10.0
    return data.to(dtype)

def run_pt_and_ov(tensors, ie_device, precision, dtype=torch.float32):
    """Run PyTorch multi_dot and OpenVINO-converted model on the same inputs."""
    class M(torch.nn.Module):
        def forward(self, *xs):
            return torch.linalg.multi_dot(xs)
    model = M().eval()
    example = tuple(tensors)
    ov_model = ov.convert_model(model, example_input=example)
    compiled = ov.Core().compile_model(
        ov_model,
        ie_device,
        config={"INFERENCE_PRECISION_HINT": str(precision)}  # e.g., "FP32" / "FP16"
    )
    with torch.no_grad():
        pt_out = model(*example).cpu().numpy()
    ov_inputs = [x.cpu().numpy() for x in example]
    ov_res = compiled(ov_inputs)[0]
    return pt_out, ov_res, ov_model

def param_sources(node, cache):
    """Recursively collect indices of original Parameters that feed into 'node'."""
    if node in cache:
        return cache[node]
    sources = set()
    if node.get_type_name() == "Parameter":
        name = node.get_friendly_name()
        try:
            idx = int(name.rsplit("_", 1)[-1])
        except Exception:
            idx = None
        if idx is not None:
            sources.add(idx)
    else:
        for inp in node.inputs():
            src_node = inp.get_source_output().get_node()
            sources |= param_sources(src_node, cache)
    cache[node] = sources
    return sources

def first_level_matmuls(ov_model):
    """Find MatMul nodes and map each to the set of source Parameter indices on its inputs."""
    order = ov_model.get_ordered_ops()
    cache = {}
    mm = []
    for n in order:
        if n.get_type_name() == "MatMul":
            left = n.input_value(0).get_node()
            right = n.input_value(1).get_node()
            ls = param_sources(left, cache)
            rs = param_sources(right, cache)
            mm.append((n, frozenset(ls), frozenset(rs)))
    return mm

def assert_close(a, b):
    np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL)

# ---------- Tests ----------
# NOTE: All tests accept 'ie_device' and 'precision' per OpenVINO layer_tests harness.

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_multi_dot_two_tensors(ie_device, precision, dtype):
    # Two tensors -> single MatMul.
    A = make_small_tensor((3, 4), dtype=dtype, seed=1)
    B = make_small_tensor((4, 2), dtype=dtype, seed=2)
    pt, ovv, _ = run_pt_and_ov([A, B], ie_device, precision, dtype=dtype)
    assert_close(pt, ovv)

@pytest.mark.parametrize("dtype", [torch.float32])
def test_multi_dot_1d_endpoints(ie_device, precision, dtype):
    # 1D endpoints -> unsqueeze to (1,N)/(N,1) and squeeze back to 1D.
    a = make_small_tensor((5,), dtype=dtype, seed=3)      # (5,)
    B = make_small_tensor((5, 7), dtype=dtype, seed=4)    # (5,7)
    c = make_small_tensor((7,), dtype=dtype, seed=5)      # (7,)
    pt, ovv, _ = run_pt_and_ov([a, B, c], ie_device, precision, dtype=dtype)
    assert (pt.ndim in (0, 1))
    assert_close(pt, ovv)

@pytest.mark.parametrize("dtype", [torch.float32])
def test_multi_dot_three_tensors_heuristic_picks_AB_then_C(ie_device, precision, dtype):
    # Three matrices: (A@B)@C is cheaper than A@(B@C).
    # A: (8x50), B: (50x6), C: (6x40)
    A = make_small_tensor((8, 50), dtype=dtype, seed=10)
    B = make_small_tensor((50, 6), dtype=dtype, seed=11)
    C = make_small_tensor((6, 40), dtype=dtype, seed=12)
    pt, ovv, ov_model = run_pt_and_ov([A, B, C], ie_device, precision, dtype=dtype)
    assert_close(pt, ovv)

    # Order check: expect an intermediate MatMul combining {0,1} before combining with {2}.
    mms = first_level_matmuls(ov_model)
    has_ab = any(({0}.issubset(L) and {1}.issubset(R)) or ({1}.issubset(L) and {0}.issubset(R))
                 for _, L, R in mms)
    assert has_ab, "Expected heuristic to compute (A@B) first"

@pytest.mark.parametrize("dtype", [torch.float32])
def test_multi_dot_three_tensors_heuristic_picks_A_then_BC(ie_device, precision, dtype):
    # Three matrices: A@(B@C) is cheaper than (A@B)@C.
    # A: (40x6), B: (6x50), C: (50x8)
    A = make_small_tensor((40, 6), dtype=dtype, seed=20)
    B = make_small_tensor((6, 50), dtype=dtype, seed=21)
    C = make_small_tensor((50, 8), dtype=dtype, seed=22)
    pt, ovv, ov_model = run_pt_and_ov([A, B, C], ie_device, precision, dtype=dtype)
    assert_close(pt, ovv)

    mms = first_level_matmuls(ov_model)
    has_bc = any(({1}.issubset(L) and {2}.issubset(R)) or ({2}.issubset(L) and {1}.issubset(R))
                 for _, L, R in mms)
    assert has_bc, "Expected heuristic to compute (B@C) first"

@pytest.mark.parametrize("dtype", [torch.float32])
def test_multi_dot_four_tensors_dp_optimal_split(ie_device, precision, dtype):
    # Four matrices: DP should choose an optimal split like (A@B) and (C@D).
    A = make_small_tensor((5, 60), dtype=dtype, seed=30)
    B = make_small_tensor((60, 3), dtype=dtype, seed=31)
    C = make_small_tensor((3, 70), dtype=dtype, seed=32)
    D = make_small_tensor((70, 4), dtype=dtype, seed=33)
    pt, ovv, ov_model = run_pt_and_ov([A, B, C, D], ie_device, precision, dtype=dtype)
    assert_close(pt, ovv)

    mms = first_level_matmuls(ov_model)
    have_ab = any((L == frozenset({0}) and R == frozenset({1})) or
                  (L == frozenset({1}) and R == frozenset({0})) for _, L, R in mms)
    have_cd = any((L == frozenset({2}) and R == frozenset({3})) or
                  (L == frozenset({3}) and R == frozenset({2})) for _, L, R in mms)
    assert have_ab and have_cd, "Expected DP to form (A@B) and (C@D) as sub-products"

@pytest.mark.parametrize("dtype", [torch.float32])
def test_multi_dot_variable_shapes_small(ie_device, precision, dtype):
    # Small/random cases (including a 1D endpoint) to cover additional paths.
    a = make_small_tensor((4,), dtype=dtype, seed=40)      # (4,)
    B = make_small_tensor((4, 3), dtype=dtype, seed=41)    # (4,3)
    C = make_small_tensor((3, 2), dtype=dtype, seed=42)    # (3,2)
    d = make_small_tensor((2,), dtype=dtype, seed=43)      # (2,)
    pt, ovv, _ = run_pt_and_ov([a, B, C, d], ie_device, precision, dtype=dtype)
    assert_close(pt, ovv)
