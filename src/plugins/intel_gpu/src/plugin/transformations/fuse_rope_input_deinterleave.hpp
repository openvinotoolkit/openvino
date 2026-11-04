// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

/// \brief Fuses RoPE input deinterleave operations into the RoPE node.
///
/// Before:
///
///                              ┌────────────────────┐
///                         ┌───►│ StridedSlice(0::2) ├───┐
///                         │    └────────────────────┘   │
///   ┌───────────────────┐ │                             │    ┌────────────────┐    ┌──────┐
///   │ Interleaved input ├─┤                             ├───►│ Concat(axis=3) ├───►│ RoPE │
///   └───────────────────┘ │    ┌────────────────────┐   │    └────────────────┘    └──────┘
///                         └───►│ StridedSlice(1::2) ├───┘
///                              └────────────────────┘
///
/// After:
///
///   ┌───────────────────┐    ┌──────────────────────────────┐
///   │ Interleaved input ├───►│ RoPE(input_interleaved=true) │
///   └───────────────────┘    └──────────────────────────────┘
///
/// The GPU RoPE kernel reads even and odd input lanes directly and produces the same
/// half-layout output as the original deinterleave subgraph. Cosine and sine inputs
/// are omitted from the graphs for clarity.
class FuseRoPEInputDeinterleave : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseRoPEInputDeinterleave");
    FuseRoPEInputDeinterleave();
};

}  // namespace ov::intel_gpu
