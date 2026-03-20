#!/usr/bin/env python3
"""Export BevPoolV2 as a *single* ONNX custom op.

Goal
- Produce an ONNX model whose graph contains exactly one node:
  `com.intel.bevpool::BevPoolV2`.

Why a dummy forward?
- `mmdet3d.ops.bev_pool_v2` calls a CUDA/C++ extension.
- For ONNX export, we only need a stable trace + a symbolic mapping.
- So `forward()` returns a zero tensor with the correct shape, and the
  real computation is delegated to the custom op at inference time.

Usage
  python export_bevpool_v2_custom_op.py \
    --out bevpool_v2_custom.onnx \
    --opset 14 \
    --N 1 --D 90 --H 54 --W 96 --C 80 \
    --out-height 128 --out-width 128 \
    --K 466560 --M 7313

The produced ONNX will require a runtime that implements the custom op
`com.intel.bevpool::BevPoolV2`.
"""

from __future__ import annotations

import argparse

import torch
from torch import nn


CUSTOM_DOMAIN = "com.intel.bevpool"
CUSTOM_OPSET_VERSION = 1


class _BevPoolV2CustomOp(torch.autograd.Function):
    """ONNX-exportable placeholder for BevPoolV2.

                Inputs are aligned with backend inference signature:
            feat:  (N, H, W, C) float
            depth: (N, D, H, W) float
            indices:   (K,) int32
            intervals: (M, 3) int32, each row is [start, end, ranks_bev]

        `intervals` layout matches `save_intervals_to_bin` in base.py:
            int3(start, end, bev_rank)

    Output:
    (N, out_channels, out_height, out_width) float (NCHW)

    Note: This forward is a placeholder; real behavior is defined by
    the ONNX custom op implementation.
    """

    @staticmethod
    def forward(
        ctx,  # noqa: ARG001
        feat: torch.Tensor,
        depth: torch.Tensor,
        indices: torch.Tensor,
        intervals: torch.Tensor,
        input_channels: int,
        output_channels: int,
        image_height: int,
        image_width: int,
        out_height: int,
        out_width: int,
    ) -> torch.Tensor:
        # Avoid Python-side shape checks during ONNX tracing/export.
        # Keep the forward as a pure placeholder tensor allocation.
        if not torch.onnx.is_in_onnx_export():
            if depth.dim() != 4:
                raise ValueError(
                    f"depth must be 4D (N,D,H,W), got {tuple(depth.shape)}")
            if feat.dim() != 4:
                raise ValueError(
                    f"feat must be 4D (N,H,W,C), got {tuple(feat.shape)}")
            if depth.size(0) != feat.size(0):
                raise ValueError(
                    f"depth/feat N mismatch: {depth.size(0)} vs {feat.size(0)}")
            if feat.size(3) != int(input_channels):
                raise ValueError(
                    f"feat C mismatch: feat.size(3)={feat.size(3)} vs input_channels={int(input_channels)}")

        # Placeholder output (NCHW) aligned with OpenVINO BevPoolV2 backend.
        # Keep explicit data dependencies on all 4 inputs to prevent exporter
        # from pruning depth/indices/intervals as unused graph inputs.
        out = feat.new_zeros((feat.size(0), int(output_channels), int(out_height), int(out_width)))
        zero = out.sum() * 0.0
        zero = zero + depth.to(dtype=out.dtype).sum() * 0.0
        zero = zero + indices.to(dtype=out.dtype).sum() * 0.0
        zero = zero + intervals.to(dtype=out.dtype).sum() * 0.0
        return out + zero

    @staticmethod
    def symbolic(
        g,
        feat,
        depth,
        indices,
        intervals,
        input_channels: int,
        output_channels: int,
        image_height: int,
        image_width: int,
        out_height: int,
        out_width: int,
    ):
        # Map to a domain-qualified custom op.
        # Consumers should register this op under domain `com.intel.bevpool`.
        out = g.op(
            f"{CUSTOM_DOMAIN}::BevPoolV2",
            feat,
            depth,
            indices,
            intervals,
            input_channels_i=int(input_channels),
            output_channels_i=int(output_channels),
            image_height_i=int(image_height),
            image_width_i=int(image_width),
            feature_height_i=int(out_height),
            feature_width_i=int(out_width),
            out_height_i=int(out_height),
            out_width_i=int(out_width),
            version_i=int(CUSTOM_OPSET_VERSION),
        )

        # Best-effort: hint output dtype + partially-known shape (NCHW).
        # This improves ONNX shape inference in some toolchains.
        try:
            out.setType(feat.type().with_sizes([None, int(output_channels), int(out_height), int(out_width)]))
        except Exception:
            pass

        return out


class BevPoolV2Module(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, image_height: int, image_width: int, out_height: int, out_width: int):
        super().__init__()
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.out_height = int(out_height)
        self.out_width = int(out_width)

    def forward(
        self,
        feat: torch.Tensor,
        depth: torch.Tensor,
        indices: torch.Tensor,
        intervals: torch.Tensor,
    ) -> torch.Tensor:
        # Important: pass out_height/out_width as Python ints so they become
        # ONNX node attributes in `symbolic()`.
        return _BevPoolV2CustomOp.apply(
            feat,
            depth,
            indices,
            intervals,
            self.input_channels,
            self.output_channels,
            self.image_height,
            self.image_width,
            self.out_height,
            self.out_width,
        )


def _make_dummy_inputs(
    *,
    n: int,
    d: int,
    h: int,
    w: int,
    c: int,
    k: int,
    m: int,
    device: torch.device,
    dtype: torch.dtype,
):
    feat = torch.rand((n, h, w, c), device=device, dtype=dtype)
    depth = torch.rand((n, d, h, w), device=device, dtype=dtype)

    # `indices` matches save_indices_to_bin input semantics in base.py:
    # a 1D index array that would be written as uint32 in bin.
    indices = torch.arange(k, device=device, dtype=torch.int32)

    # `intervals` matches save_intervals_to_bin semantics in base.py:
    # each row is [start, end, ranks_bev] (int3).
    # Here we split K points into M contiguous intervals.
    m = max(1, min(int(m), int(k)))
    base_len = k // m
    rem = k % m
    lengths_list = [base_len + (1 if idx < rem else 0) for idx in range(m)]

    interval_lengths = torch.tensor(lengths_list, device=device, dtype=torch.int32)
    interval_starts = torch.zeros((m,), device=device, dtype=torch.int32)
    if m > 1:
        interval_starts[1:] = torch.cumsum(interval_lengths[:-1], dim=0)
    interval_ends = interval_starts + interval_lengths
    ranks_bev = torch.arange(m, device=device, dtype=torch.int32)
    intervals = torch.stack((interval_starts, interval_ends, ranks_bev), dim=1)

    return feat, depth, indices, intervals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="bevpool_v2_custom.onnx")
    parser.add_argument("--opset", type=int, default=14)

    parser.add_argument("--N", type=int, default=1)
    parser.add_argument("--D", type=int, default=90)
    parser.add_argument("--H", type=int, default=54)
    parser.add_argument("--W", type=int, default=96)
    parser.add_argument("--C", type=int, default=80)
    parser.add_argument("--K", type=int, default=466560)
    parser.add_argument("--M", type=int, default=7313)

    parser.add_argument("--out-height", type=int, default=128)
    parser.add_argument("--out-width", type=int, default=128)

    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--fp16", action="store_true")

    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float16 if args.fp16 else torch.float32

    model = BevPoolV2Module(input_channels=args.C,
                            output_channels=1,
                            image_height=args.H,
                            image_width=args.W,
                            out_height=args.out_height,
                            out_width=args.out_width).eval().to(device)

    inputs = _make_dummy_inputs(
        n=args.N,
        d=args.D,
        h=args.H,
        w=args.W,
        c=args.C,
        k=args.K,
        m=args.M,
        device=device,
        dtype=dtype,
    )

    input_names = [
        "feat",
        "depth",
        "indices",
        "intervals",
    ]
    output_names = ["bev_feat"]

    dynamic_axes = {
        "depth": {0: "N", 1: "D", 2: "H", 3: "W"},
        "feat": {0: "N", 1: "H", 2: "W", 3: "C"},
        "indices": {0: "K"},
        "intervals": {0: "M"},
        "bev_feat": {0: "N", 1: "out_channels", 2: "out_height", 3: "out_width"},
    }

    # Export.
    torch.onnx.export(
        model,
        inputs,
        args.out,
        dynamo=False,
        opset_version=int(args.opset),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        custom_opsets={CUSTOM_DOMAIN: CUSTOM_OPSET_VERSION},
        do_constant_folding=False,
    )

    print(f"[OK] Exported: {args.out}")
    print(f"     Custom domain: {CUSTOM_DOMAIN} (version {CUSTOM_OPSET_VERSION})")


if __name__ == "__main__":
    main()
