// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov::intel_gpu {

// Keeps xattention threshold input in fp32 by marking it as precision-sensitive
// before ConvertPrecision pass. This avoids down-conversion to fp16 which can
// make thresholding unstable at certain boundary values.
class KeepXAttentionThresholdPrecision : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("KeepXAttentionThresholdPrecision");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::intel_gpu
