// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov::intel_gpu {

// Keeps xattention threshold input in fp32 by marking it as precision-sensitive
// before ConvertPrecision pass. This avoids down-conversion to fp16 which can
// make thresholding unstable at certain boundary values.
class KeepXAttentionThresholdPrecision : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::intel_gpu::KeepXAttentionThresholdPrecision");
    KeepXAttentionThresholdPrecision();
};


}  // namespace ov::intel_gpu
