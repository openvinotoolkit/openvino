# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from openvino import Core, PartialShape, convert_model
import openvino.properties.hint as hints
from openvino import Type


def window_partition(x, window_size):
    # x: (B, H, W, C) -> (num_windows*B, window_size, window_size, C)
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    # Mirrors SwinIR/Swin window_reverse: B is computed via int(...), which a tracer
    # bakes into a literal constant in the view shape -- the source of the batch bug.
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowReverseModel(torch.nn.Module):
    """A round-trip through window_partition/window_reverse, exactly the structure that
    produces the frozen-batch view shapes in SwinIR. The identity round-trip lets us
    compare against a trivially-correct reference at any batch size.

    H and W are read from the input tensor at runtime (as SwinIR does), so under tracing
    the spatial dims become dynamic `aten::size` expressions while the batch B collapses
    to a literal via `int(...)` -- reproducing the frozen-batch / dynamic-spatial Concat
    that ReshapeBatchDimResolver must repair.

    A `Linear(c_in, embed_dim)` projects to a fixed channel width before the window ops,
    mirroring real SwinIR (which carries a static embed_dim=180 through the window blocks).
    This makes the window_reverse channel STATIC -- the dimension the resolver recovers by
    walking back from the view's data -- so the test exercises the same path as the model."""

    def __init__(self, window_size, c_in, embed_dim):
        super().__init__()
        self.window_size = window_size
        self.proj = torch.nn.Linear(c_in, embed_dim)

    def forward(self, x):
        # x: (B, C_in, H, W) -> (B, H, W, C_in); H, W flow from the dynamic input shape.
        H = x.shape[2]
        W = x.shape[3]
        x = x.permute(0, 2, 3, 1).contiguous()
        # Project to a fixed embed_dim so the window channel is static, as in SwinIR.
        x = self.proj(x)
        windows = window_partition(x, self.window_size)
        # touch the windows so the graph is non-trivial but still an identity overall
        windows = windows + 0.0
        x = window_reverse(windows, self.window_size, H, W)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class TestWindowReverseBatchReshape:
    """Regression for the PyTorch-FE baked-batch reshape bug.

    A model traced at batch=1 must still produce correct results after the converted
    model is reshaped to batch=2 (the SwinIR_Classical E2E failure). Before the
    ReshapeBatchDimResolver pass the doubled batch was absorbed into the window_reverse
    channel and the output was scrambled (max_abs_diff ~3.47); after the fix it tracks
    the real batch.
    """

    @staticmethod
    def _convert_batch1(window_size, H, W, C, embed_dim):
        model = WindowReverseModel(window_size, C, embed_dim).eval()
        example = torch.randn(1, C, H, W)
        with torch.no_grad():
            om = convert_model(model, example_input=example)
        return model, om

    @pytest.mark.parametrize("window_size,H,W,C,embed_dim", [(8, 16, 16, 6, 12), (4, 12, 8, 3, 8)])
    @pytest.mark.parametrize("batch", [1, 2, 3])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_window_reverse_rebatch(self, window_size, H, W, C, embed_dim, batch, ie_device, precision):
        model, om = self._convert_batch1(window_size, H, W, C, embed_dim)

        # Reshape the converted (traced-at-batch-1) model to the target batch.
        om.reshape({om.input(0): PartialShape([batch, C, H, W])})

        np_in = np.random.randn(batch, C, H, W).astype(np.float32)
        with torch.no_grad():
            ref = model(torch.from_numpy(np_in)).numpy()

        config = {}
        if precision == "FP32":
            config[hints.inference_precision] = Type.f32
        compiled = Core().compile_model(om, ie_device, config)
        ov_out = np.asarray(compiled(np_in)[compiled.output(0)], dtype=np.float32)

        eps = 1e-4 if precision == "FP32" else 5e-2
        assert ov_out.shape == ref.shape, f"shape mismatch ov={ov_out.shape} ref={ref.shape}"
        n_bad = int((~np.isclose(ov_out, ref, atol=eps, rtol=eps)).sum())
        max_abs = float(np.abs(ov_out - ref).max())
        assert n_bad == 0, f"accuracy failed: {n_bad} mismatches, max_abs_diff={max_abs}"

    @pytest.mark.parametrize("window_size,H,W,C,embed_dim", [(8, 16, 16, 6, 12)])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_window_reverse_pass_fired(self, window_size, H, W, C, embed_dim, ie_device, precision):
        """Structural check: ReshapeBatchDimResolver must rebuild every window_reverse
        view shape so no Reshape keeps a frozen positive-int leading constant alongside a
        single trailing -1 (the baked-batch signature). Device/precision-independent
        (ie_device/precision are accepted only to satisfy the shared parametrization)."""
        _, om = self._convert_batch1(window_size, H, W, C, embed_dim)

        offenders = []
        for node in om.get_ordered_ops():
            if node.get_type_info().name != "Reshape":
                continue
            shape_src = node.input_value(1).get_node()
            if shape_src.get_type_info().name != "Concat":
                continue
            elems = [shape_src.input_value(i).get_node() for i in range(len(shape_src.inputs()))]

            def const_scalar(n):
                if n.get_type_info().name != "Constant":
                    return None
                vals = n.data.flatten()
                return int(vals[0]) if vals.size == 1 else None

            lead = const_scalar(elems[0])
            neg1_positions = [i for i, e in enumerate(elems) if const_scalar(e) == -1]
            # The offending pattern: positive-int leading constant + exactly one trailing -1.
            if lead is not None and lead > 0 and neg1_positions == [len(elems) - 1]:
                offenders.append(node.get_friendly_name())

        assert not offenders, f"baked-batch reshapes not rewritten: {offenders}"


class TestPlainReshapeNotAffected:
    """Negative test: ordinary reshapes (no -1, or -1 not in the channel position, or a
    fully-static shape vector) must be left untouched and stay correct after re-batching,
    proving ReshapeBatchDimResolver does not fire on legitimate leading-constant reshapes."""

    class PlainReshape(torch.nn.Module):
        def forward(self, x):
            # leading 1 is a genuine literal; shape is fully static, no -1 -> must NOT match.
            b, c, h, w = x.shape
            y = x.reshape(1, c * h * w)
            return y.reshape(b, c, h, w)

    @pytest.mark.parametrize("batch", [1, 2])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_plain_reshape_rebatch(self, batch, ie_device, precision):
        model = self.PlainReshape().eval()
        example = torch.randn(1, 3, 4, 5)
        with torch.no_grad():
            om = convert_model(model, example_input=example)
        # This model is not batch-reshapeable (the middle reshape pins total size); just
        # verify batch=1 stays correct and conversion is untouched by the pass.
        np_in = np.random.randn(1, 3, 4, 5).astype(np.float32)
        with torch.no_grad():
            ref = model(torch.from_numpy(np_in)).numpy()
        config = {}
        if precision == "FP32":
            config[hints.inference_precision] = Type.f32
        compiled = Core().compile_model(om, ie_device, config)
        ov_out = np.asarray(compiled(np_in)[compiled.output(0)], dtype=np.float32)
        eps = 1e-4 if precision == "FP32" else 5e-2
        assert np.isclose(ov_out, ref, atol=eps, rtol=eps).all()


def _has_rewritten_reshape(om):
    """Return True if any Reshape in `om` was rewritten by ReshapeBatchDimResolver. The pass
    turns the leading baked-batch Constant into a Constant(-1) and pins the former trailing -1
    channel to a Constant holding data's (static) last dimension, i.e. a positive value. So the
    fingerprint is a shape Concat whose leading element is Constant(-1) and whose last element is
    a positive Constant -- the inverse of the offending baked pattern."""
    for node in om.get_ordered_ops():
        if node.get_type_info().name != "Reshape":
            continue
        shape_src = node.input_value(1).get_node()
        if shape_src.get_type_info().name != "Concat":
            continue
        elems = [shape_src.input_value(i).get_node() for i in range(len(shape_src.inputs()))]
        if len(elems) < 2:
            continue

        def const_scalar(n):
            if n.get_type_info().name != "Constant":
                return None
            vals = n.data.flatten()
            return int(vals[0]) if vals.size == 1 else None

        lead = const_scalar(elems[0])
        last = const_scalar(elems[-1])
        if lead == -1 and last is not None and last > 0:
            return True
    return False


class TestReshapeFalsePositiveNotFired:
    """Negative tests for ReshapeBatchDimResolver over-matching.

    These models contain the exact structural signature the pass keys on -- a literal leading
    batch constant, a dynamic interior dim from the input shape, and a single trailing -1 built
    as a Concat under TorchScript tracing. But the -1 here means "product of the remaining
    dimensions" (H*W, or heads*head_dim), which is NOT the data tensor's last dimension. The
    earlier (loose) pass rewrote the channel to Gather(ShapeOf(data), -1) and corrupted the output
    even at the traced batch=1. The hardened predicate must leave these untouched: data's last
    dimension is dynamic (it comes straight from a Parameter), so the static-last-dim gate excludes
    them. We assert BOTH that the pass did not rewrite the Reshape (structural) and that batch=1
    output matches torch (numeric -- catches the traced-batch corruption)."""

    class SpatialFlatten(torch.nn.Module):
        # x: (B, C, H, W); reshape(1, C, -1) -> -1 == H*W, NOT data's last dim (W).
        def forward(self, x):
            C = x.shape[1]
            return x.reshape(1, C, -1)

    class AttnHeadMerge(torch.nn.Module):
        # x: (1, heads, N, head_dim); after transpose, reshape(1, N, -1) -> -1 == heads*head_dim,
        # NOT data's last dim (head_dim). The classic attention output merge.
        def forward(self, x):
            N = x.shape[2]
            x = x.transpose(1, 2)
            return x.reshape(1, N, -1)

    @pytest.mark.parametrize(
        "model_cls,shape",
        [
            (SpatialFlatten, (1, 3, 4, 5)),
            (AttnHeadMerge, (1, 4, 6, 8)),
        ],
    )
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_false_positive_not_fired(self, model_cls, shape, ie_device, precision):
        model = model_cls().eval()
        example = torch.randn(*shape)
        with torch.no_grad():
            om = convert_model(model, example_input=example)

        # Structural: the pass must NOT have rewritten this ordinary reshape.
        assert not _has_rewritten_reshape(om), (
            f"{model_cls.__name__}: ReshapeBatchDimResolver fired on an ordinary reshape"
        )

        # Numeric at the traced batch: the loose pass corrupted the result even here.
        np_in = np.random.randn(*shape).astype(np.float32)
        with torch.no_grad():
            ref = model(torch.from_numpy(np_in)).numpy()
        config = {}
        if precision == "FP32":
            config[hints.inference_precision] = Type.f32
        compiled = Core().compile_model(om, ie_device, config)
        ov_out = np.asarray(compiled(np_in)[compiled.output(0)], dtype=np.float32)
        eps = 1e-4 if precision == "FP32" else 5e-2
        assert ov_out.shape == ref.shape, f"shape mismatch ov={ov_out.shape} ref={ref.shape}"
        assert np.isclose(ov_out, ref, atol=eps, rtol=eps).all()
