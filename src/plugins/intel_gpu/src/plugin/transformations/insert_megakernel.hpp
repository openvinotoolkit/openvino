// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov::intel_gpu {

/// Replaces all 28 transformer-decoder layers of Qwen3-0.6B
class InsertMegaKernel : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("InsertMegaKernel");
    InsertMegaKernel() = default;
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

}  // namespace ov::intel_gpu
