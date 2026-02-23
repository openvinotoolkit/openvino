// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"

namespace ov::intel_gpu {

class IncreaseRMSInputPrecision: public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("IncreaseRMSInputPrecision");
    IncreaseRMSInputPrecision();
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

class IncreasePrecisionForQwenVLMerger: public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("IncreasePrecisionForQwenVLMerger");
    IncreasePrecisionForQwenVLMerger();
};


}   // namespace ov::intel_gpu