// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

/// @brief Absorbs a Transpose({0,2,1,3}) following an SDPA into the SDPA's
/// output_transpose_order.
///
/// When an SDPA op is followed by a Transpose that swaps the heads and sequence
/// dimensions (axes 1 and 2 in a 4-D [b,h,seq,d] tensor), the Transpose can be
/// eliminated by composing it with the SDPA's output_transpose_order:
///
///   SDPA(out_order) → Transpose({0,2,1,3})
///      becomes
///   SDPA(out_order ∘ {0,2,1,3})
///
/// A second matcher handles framework v13::ScaledDotProductAttention ops that
/// were never converted to the internal op::SDPA (e.g. TransposeSDPAMatcher
/// bails when an input transpose moves the head_size dim, such as K preceded
/// by Transpose({0,1,3,2}) in pi05). Those are converted to op::SDPA with
/// identity input orders, absorbing only the output Transpose. (The input
/// transposes are intentionally left standalone: absorbing a head_size-moving
/// transpose into the SDPA is correct but slower, as the kernel would then read
/// the input with a strided, non-contiguous head_size access pattern.)
///
/// This avoids a separate Permute kernel on GPU without changing the numerical
/// result.
class SDPATransposeFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("SDPATransposeFusion");
    SDPATransposeFusion();
};

}  // namespace ov::intel_gpu
