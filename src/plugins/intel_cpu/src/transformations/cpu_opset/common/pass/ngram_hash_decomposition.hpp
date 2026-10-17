// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov::intel_cpu {

// Rewrites `offsets + (xor_i(token_i * mult_i) mod moduli)` (traced e.g. from a token n-gram hash) into
// equivalent int32/f32 arithmetic on 12-bit limbs, for the case where `mult_i` is a Constant that doesn't
// fit int32: left as int64, ConvertPrecision clamps `mult_i` to INT32_MAX and the result is silently wrong.
// Only engages when some multiplier is out of i32 range; otherwise leaves the graph untouched.
class NgramHashDecomposition : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("NgramHashDecomposition");
    NgramHashDecomposition();
};

}  // namespace ov::intel_cpu
