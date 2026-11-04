// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov::intel_cpu {

/**
 * @brief Saturates f32 scalar constants of Multiply/Add/Subtract that exceed the bf16 range (e.g. FLT_MAX from
 *        attention masks), which would otherwise overflow to inf in bf16.
 */
class ClampBF16ScalarConstants : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ClampBF16ScalarConstants");
    ClampBF16ScalarConstants();
};

}  // namespace ov::intel_cpu
