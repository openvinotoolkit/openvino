// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov::intel_cpu {

/**
 * @brief Saturates f32 scalar constants of Multiply/Add/Subtract that exceed the bf16 range, which would otherwise
 *        overflow to inf when the op runs in bf16.
 *
 * E.g. an attention mask `(1 - mask) * f32_lowest` is reassociated into `mask * f32_max + f32_lowest`, and in bf16
 * `mask * f32_max` becomes inf, so Softmax later produces NaN:
 *
 *   before:  mask -> Multiply(f32_max) -> Add(f32_lowest) -> ...
 *   after:   mask -> Multiply(bf16_max) -> Add(-bf16_max) -> ...
 *
 * Restrictions:
 *  - Multiply/Add/Subtract only: the eltwise ops that mask-building chains consist of, and the ones that
 *    ConvertToPowerStatic and Snippets pick up afterwards.
 *  - Scalar constants only: no overflow with non-scalar ones has been reproduced.
 *  - Must run before Snippets tokenization and ConvertToPowerStatic, which would otherwise consume the constants.
 */
class ClampBF16ScalarConstants : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ClampBF16ScalarConstants");
    ClampBF16ScalarConstants();
};

}  // namespace ov::intel_cpu
