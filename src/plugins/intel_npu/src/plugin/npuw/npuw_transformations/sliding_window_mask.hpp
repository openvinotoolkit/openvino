// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov::npuw {

// Fixes sliding window attention mask for static KV buffers in LLM models.
// Supports Phi-3 (transformers 4.51 & 4.53) and Gemma4 patterns.
class SlidingWindowMask : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::SlidingWindowMask");
    SlidingWindowMask() = default;
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::npuw
