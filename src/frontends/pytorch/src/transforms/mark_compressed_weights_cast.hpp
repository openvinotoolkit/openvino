// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov::frontend::pytorch::pass {

/**
 * @interface MarkCompressedWeightsCast
 * @brief Marks the trailing precision cast of a weight-decompression subgraph as decompression.
 *
 * E.g. Constant(i4) -> Convert(f16) -> [Subtract] -> Multiply -> Convert(bf16) in a bf16 model: no other pass
 * marks that cast, so consumers gating on ov::is_decompression() (CPU ConvertMatMulToFC) would reject the weights.
 */
class MarkCompressedWeightsCast : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::pytorch::pass::MarkCompressedWeightsCast");
    MarkCompressedWeightsCast();
};

}  // namespace ov::frontend::pytorch::pass
