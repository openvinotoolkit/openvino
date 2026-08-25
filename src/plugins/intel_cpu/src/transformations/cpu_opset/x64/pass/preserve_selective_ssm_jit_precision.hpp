// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov::intel_cpu {

/// Prevents generic CPU precision conversion from widening SelectiveSSM operations accepted by the x64 JIT executor.
class PreserveSelectiveSSMJitPrecision final : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("PreserveSelectiveSSMJitPrecision");

    PreserveSelectiveSSMJitPrecision();
};

}  // namespace ov::intel_cpu
