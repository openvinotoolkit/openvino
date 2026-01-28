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
    data = (torch.randint(-100, 101, shape, generator=g, device=device, dtype=torch.int32).to(torch.float32)) / 10.0
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
    # 🇮🇱 שני טנזורים – MatMul בודד.
    A = make_small_tensor((3, 4), dtype=dtype, seed=1)
    B = make_small_tensor((4, 2), dtype=dtype, seed=2)
    pt, ovv, _ = run_pt_and_ov([A, B], ie_device, precision, dtype=dtype)
    assert_close(pt, ovv)

@pytest.mark.parametrize("dtype", [torch.float32])
def test_multi_dot_1d_endpoints(ie_device, precision, dtype):
    # 🇮🇱 קצוות 1D – Unsqueeze ל-(1,N)/(N,1) ואז Squeeze חזרה ל-1D.
    a = make_small_tensor((5,), dtype=dtype, seed=3)      # (5,)
    B = make_small_tensor((5, 7), dtype=dtype, seed=4)    # (5,7)
    c = make_small_tensor((7,), dtype=dtype, seed=5)      # (7,)
    pt, ovv, _ = run_pt_and_ov([a, B, c], ie_device, precision, dtype=dtype)
    assert (pt.ndim in (0, 1))
    assert_close(pt, ovv)

@pytest.mark.parametrize("dtype", [torch.float32])
def test_multi_dot_three_tensors_heuristic_picks_AB_then_C(ie_device, precision, dtype):
    # 🇮🇱 שלוש מטריצות – מקרה שבו זול יותר לחשב (A@B)@C מאשר A@(B@C).
    # A: (8x50), B: (50x6), C: (6x40)
    A = make_small_tensor((8, 50), dtype=dtype, seed=10)
    B = make_small_tensor((50, 6), dtype=dtype, seed=11)
    C = make_small_tensor((6, 40), dtype=dtype, seed=12)
    pt, ovv, ov_model = run_pt_and_ov([A, B, C], ie_device, precision, dtype=dtype)
    assert_close(pt, ovv)

    # 🇮🇱 בדיקת סדר: מצפים ל-MatMul ביניים שמחבר {0,1} לפני שילוב עם {2}.
    mms = first_level_matmuls(ov_model)
    has_ab = any(({0}.issubset(L) and {1}.issubset(R)) or ({1}.issubset(L) and {0}.issubset(R))
                 for _, L, R in mms)
    assert has_ab, "Expected heuristic to compute (A@B) first"

@pytest.mark.parametrize("dtype", [torch.float32])
def test_multi_dot_three_tensors_heuristic_picks_A_then_BC(ie_device, precision, dtype):
    # 🇮🇱 שלוש מטריצות – מקרה שבו זול יותר A@(B@C) מאשר (A@B)@C.
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
    # 🇮🇱 ארבע מטריצות – בדיקת DP עם פיצול אופטימלי ((A@B)@(C@D)).
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
    # 🇮🇱 מקרים קטנים/רנדומליים (כולל 1D בקצה אחד), כדי לכסות מסלולי קוד שונים.
    a = make_small_tensor((4,), dtype=dtype, seed=40)      # (4,)
    B = make_small_tensor((4, 3), dtype=dtype, seed=41)    # (4,3)
    C = make_small_tensor((3, 2), dtype=dtype, seed=42)    # (3,2)
    d = make_small_tensor((2,), dtype=dtype, seed=43)      # (2,)
    pt, ovv, _ = run_pt_and_ov([a, B, C, d], ie_device, precision, dtype=dtype)
    assert_close(pt, ovv)
import numpy as np
import torch
import openvino as ov

RTOL_MDOT = 1e-4
ATOL_MDOT = 1e-4


def make_small_tensor(shape, dtype=torch.float32, device="cpu", seed=0):
    """Create a small deterministic tensor in [-10, 10] with one decimal digit."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    data = (
        torch.randint(-100, 101, shape, generator=g, device=device, dtype=torch.int32)
        .to(torch.float32)
        / 10.0
    )
    return data.to(dtype)


def run_pt_and_ov_multi_dot(tensors, ie_device, precision, dtype=torch.float32):
    """Run PyTorch torch.linalg.multi_dot and the OpenVINO-converted model on the same inputs."""

    class MultiDotWrapper(torch.nn.Module):
        def forward(self, *xs):
            return torch.linalg.multi_dot(xs)

    model = MultiDotWrapper().eval()
    example = tuple(tensors)

    # Use the robust wrapper that works with/without openvino.convert_model
    ov_model = ov_convert(model, example_input=example)

    core = ov.Core()
    compiled = core.compile_model(
        ov_model,
        ie_device,
        config={"INFERENCE_PRECISION_HINT": str(precision)},
    )

    with torch.no_grad():
        pt_out = model(*example).detach().cpu().numpy()

    ov_inputs = [x.detach().cpu().numpy() for x in example]
    ov_res_map = compiled.create_infer_request().infer(ov_inputs)
    # Single-output model -> take first value
    ov_out = next(iter(ov_res_map.values()))

    return pt_out, ov_out, ov_model


def assert_close_multi_dot(a, b):
    """Helper for numerical comparison with multi_dot-specific tolerances."""
    np.testing.assert_allclose(a, b, rtol=RTOL_MDOT, atol=ATOL_MDOT)


def _param_sources(node, cache):
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
            sources |= _param_sources(src_node, cache)

    cache[node] = sources
    return sources


def collect_matmul_param_sets(ov_model):
    """
    Collect all MatMul nodes and, for each one, the sets of source Parameter indices
    on its left and right inputs.

    Returns:
        List of tuples: (matmul_node, left_sources, right_sources)
        where left_sources/right_sources are frozenset[int] of parameter indices.
    """
    order = ov_model.get_ordered_ops()
    cache = {}
    mm = []
    for n in order:
        if n.get_type_name() == "MatMul":
            left_node = n.input_value(0).get_node()
            right_node = n.input_value(1).get_node()
            ls = _param_sources(left_node, cache)
            rs = _param_sources(right_node, cache)
            mm.append((n, frozenset(ls), frozenset(rs)))
    return mm
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_multi_dot_three_tensors_prefers_AB_then_C(ie_device, precision, ir_version, dtype):
    """
    Three matrices case where (A @ B) @ C is cheaper than A @ (B @ C).
    We expect the translator's heuristic to form the AB product first.
    """
    # Shapes:
    #   A: (8, 50)
    #   B: (50, 6)
    #   C: (6, 40)
    # Cost((A @ B) @ C) = 8*50*6 + 8*6*40 = 2400 + 1920 = 4320
    # Cost(A @ (B @ C)) = 50*6*40 + 8*50*40 = 12000 + 16000 = 28000
    A = make_small_tensor((8, 50), dtype=dtype, seed=100)
    B = make_small_tensor((50, 6), dtype=dtype, seed=101)
    C = make_small_tensor((6, 40), dtype=dtype, seed=102)

    pt_out, ov_out, ov_model = run_pt_and_ov_multi_dot(
        [A, B, C],
        ie_device=ie_device,
        precision=precision,
        dtype=dtype,
    )
    assert_close_multi_dot(pt_out, ov_out)

    # Check structure: there must exist a MatMul whose inputs come exactly from {0,1}
    # (i.e. it forms AB as a sub-product before combining with C).
    matmuls = collect_matmul_param_sets(ov_model)
    has_ab = any(
        (L == frozenset({0}) and R == frozenset({1}))
        or (L == frozenset({1}) and R == frozenset({0}))
        for _, L, R in matmuls
    )
    assert has_ab, "Expected heuristic to compute (A @ B) as an intermediate sub-product"


@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_multi_dot_three_tensors_prefers_A_then_BC(ie_device, precision, ir_version, dtype):
    """
    Three matrices case where A @ (B @ C) is cheaper than (A @ B) @ C.
    We expect the translator's heuristic to form the BC product first.
    """
    # Shapes:
    #   A: (40, 6)
    #   B: (6, 50)
    #   C: (50, 8)
    # Cost((A @ B) @ C) = 40*6*50 + 40*50*8 = 12000 + 16000 = 28000
    # Cost(A @ (B @ C)) = 6*50*8 + 40*6*8 = 2400 + 1920 = 4320
    A = make_small_tensor((40, 6), dtype=dtype, seed=200)
    B = make_small_tensor((6, 50), dtype=dtype, seed=201)
    C = make_small_tensor((50, 8), dtype=dtype, seed=202)

    pt_out, ov_out, ov_model = run_pt_and_ov_multi_dot(
        [A, B, C],
        ie_device=ie_device,
        precision=precision,
        dtype=dtype,
    )
    assert_close_multi_dot(pt_out, ov_out)

    # Check structure: there must exist a MatMul whose inputs come exactly from {1,2}
    # (i.e. it forms BC as a sub-product before combining with A).
    matmuls = collect_matmul_param_sets(ov_model)
    has_bc = any(
        (L == frozenset({1}) and R == frozenset({2}))
        or (L == frozenset({2}) and R == frozenset({1}))
        for _, L, R in matmuls
    )
    assert has_bc, "Expected heuristic to compute (B @ C) as an intermediate sub-product"


@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_multi_dot_four_tensors_dp_forms_AB_and_CD(ie_device, precision, ir_version, dtype):
    """
    Four matrices case where the DP-style splitter should form (A @ B) and (C @ D)
    as independent sub-products before the final MatMul.
    """
    # Shapes chosen so that ((A @ B) @ (C @ D)) is naturally optimal for a DP planner.
    #   A: (5, 60)
    #   B: (60, 3)
    #   C: (3, 70)
    #   D: (70, 4)
    A = make_small_tensor((5, 60), dtype=dtype, seed=300)
    B = make_small_tensor((60, 3), dtype=dtype, seed=301)
    C = make_small_tensor((3, 70), dtype=dtype, seed=302)
    D = make_small_tensor((70, 4), dtype=dtype, seed=303)

    pt_out, ov_out, ov_model = run_pt_and_ov_multi_dot(
        [A, B, C, D],
        ie_device=ie_device,
        precision=precision,
        dtype=dtype,
    )
    assert_close_multi_dot(pt_out, ov_out)

    matmuls = collect_matmul_param_sets(ov_model)

    # Look for explicit (A @ B): parameters {0} and {1}
    have_ab = any(
        (L == frozenset({0}) and R == frozenset({1}))
        or (L == frozenset({1}) and R == frozenset({0}))
        for _, L, R in matmuls
    )

    # Look for explicit (C @ D): parameters {2} and {3}
    have_cd = any(
        (L == frozenset({2}) and R == frozenset({3}))
        or (L == frozenset({3}) and R == frozenset({2}))
        for _, L, R in matmuls
    )

    assert have_ab and have_cd, "Expected DP to form (A @ B) and (C @ D) as sub-products"


@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_multi_dot_matmul_count_is_n_minus_1(ie_device, precision, ir_version, dtype):
    """
    Sanity check: for N tensors, the resulting graph should contain at least N-1 MatMul nodes.
    This ensures no degenerate lowering that would expand into something more expensive.
    """
    # Example with 5 tensors in a valid chain:
    A = make_small_tensor((4, 10), dtype=dtype, seed=400)
    B = make_small_tensor((10, 6), dtype=dtype, seed=401)
    C = make_small_tensor((6, 3), dtype=dtype, seed=402)
    D = make_small_tensor((3, 8), dtype=dtype, seed=403)
    E = make_small_tensor((8, 2), dtype=dtype, seed=404)

    tensors = [A, B, C, D, E]
    pt_out, ov_out, ov_model = run_pt_and_ov_multi_dot(
        tensors,
        ie_device=ie_device,
        precision=precision,
        dtype=dtype,
    )
    assert_close_multi_dot(pt_out, ov_out)

    matmuls = [n for n in ov_model.get_ordered_ops() if n.get_type_name() == "MatMul"]
    assert len(matmuls) >= len(tensors) - 1, (
        f"Expected at least {len(tensors) - 1} MatMul nodes for "
        f"{len(tensors)}-tensor multi_dot chain, got {len(matmuls)}"
    )
