// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov::intel_cpu {

// Fuses an int64 `offsets + floor_mod(xor_i(token_i * mult_i), moduli)` hash (traced e.g. from a token n-gram
// hash) into a single NgramHash node that keeps the int64 arithmetic inside its kernel. Left as separate ops,
// ConvertPrecision downcasts the whole subgraph to i32 and the products/xors silently wrap at 32 bits.
class NgramHashFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("NgramHashFusion");
    NgramHashFusion();
};

}  // namespace ov::intel_cpu
