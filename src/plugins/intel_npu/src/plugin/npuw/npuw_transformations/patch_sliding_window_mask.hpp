// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov::npuw {

// Public API for patching sliding window attention masks.
// Applies fixes for Phi-3, Gemma4, and other models with sliding window attention.
class PatchSlidingWindowMask : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::PatchSlidingWindowMask");
    explicit PatchSlidingWindowMask() = default;
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::npuw
