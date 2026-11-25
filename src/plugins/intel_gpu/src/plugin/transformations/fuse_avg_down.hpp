// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

/// \brief Replaces AvgDown subgraphs (space-to-depth followed by a channel-group mean) with GPU-friendly operations.
///
/// Before:
///
///   Input                       [B, C, T, H, W]
///     │
///   Pad (optional)              zero padding at the front of T up to a multiple of ft
///     │
///   Reshape                     [B, C, T', ft, H', fs, W', fs]
///     │
///   Transpose                   order [0, 1, 3, 5, 7, 2, 4, 6]
///     │
///   Reshape (optional)          [B, C*ft*fs*fs, T', H', W']  (flatten; usually merged into the next Reshape)
///     │
///   Reshape                     [B, C_out, g, T', H', W']
///     │
///   ReduceMean                  axes = {2} (or {-4}), keep_dims = false  ->  [B, C_out, T', H', W']
///
/// After, depending on the factors (ft = temporal factor, fs = spatial factor, g = group size):
///
///   ft=1, fs=2, C_out=C, g=4 (spatial downsample):
///   ┌─────────┐     ┌──────────────────────────────────────────┐
///   │  Input  ├────►│ AvgPool(kernel=[1,2,2], strides=[1,2,2]) │
///   └─────────┘     └──────────────────────────────────────────┘
///
///   ft=2, fs=2, g=4 (temporal/spatial downsample):
///   ┌─────────┐     ┌─────────────────────┐
///   │  Input  ├────►│ GroupedSpaceToDepth │
///   └─────────┘     └─────────────────────┘
///
///   ft=1, fs=1, C_out=C, g=1 (identity downsample):
///   ┌─────────┐     ┌─────────┐
///   │  Input  ├────►│ Output  │
///   └─────────┘     └─────────┘
///
/// Conditions:
///  - The Reshapes and the Transpose have a single consumer.
///  - The flatten Reshape may be absent; earlier graph optimizations usually merge it with the 6-D Reshape.
///  - Static shapes: the factors are taken from the inferred 8-D and 6-D Reshape outputs, which must be consistent
///    with the (padded) input B,C,T,H,W. special_zero on the Reshapes is accepted.
///  - Dynamic shapes: the factors are taken from the Concat target-shape expressions, which must share the
///    ShapeOf-derived T', H', W' parts; this needs the flatten Reshape and no special_zero Reshape.
///  - The optional Pad is CONSTANT with zero value and only pads the front of the time axis to a multiple of ft;
///    it is folded into the replacement.
///  - input_channels * ft * fs * fs == C_out * g, and the replacement output shape is compatible with the ReduceMean output.
class FuseAvgDown : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseAvgDown");
    FuseAvgDown();
};

}  // namespace ov::intel_gpu
