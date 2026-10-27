// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::npuw {

// Rewrites the LM-head dequantize chain emitted by the WebNN / ORT export into the layout
// the NPUW tail-vocab matchers (PreserveConstDictMatMulAsymm) understand, i.e. the one
// optimum-intel produces.
//
// Before:
//   Const(W:u8 [OC,1,IC]) -> Convert ------------------------------+
//   Const(Z:u8 [OC])      -> [Reshape ->] Convert -> [Reshape ->]  -+-> Subtract
//   Const(S:f16 [OC])     -> [Reshape ->] ---------------------------> Multiply
//     -> Reshape [OC,IC] -> Transpose [IC,OC] -> MatMul(transpose_b=false)
//
// After:
//   Const(W:u8 [OC,IC]) -> Convert ------------+
//   Const(Z:u8 [OC,1])  -> Convert -------------> Subtract -> Multiply(Const(S:f16 [OC,1]))
//     -> MatMul(transpose_b=true)
//
// All Reshapes are applied to the (still compressed) Constants, so the weight stays u8 and
// no data is copied. The pass is a no-op for any graph that doesn't match this chain exactly.
class CanonicalizeLMHeadDQLayout : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::npuw::CanonicalizeLMHeadDQLayout");
    CanonicalizeLMHeadDQLayout();
};

}  // namespace ov::npuw
