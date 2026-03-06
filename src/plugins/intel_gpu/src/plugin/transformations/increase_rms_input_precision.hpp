// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"

namespace ov::intel_gpu {

class IncreaseRMSInputPrecision: public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("IncreaseRMSInputPrecision");
    explicit IncreaseRMSInputPrecision(bool use_onednn = false);
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    bool m_use_onednn;
};

class IncreasePrecisionForQwenVLMerger: public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("IncreasePrecisionForQwenVLMerger");
    IncreasePrecisionForQwenVLMerger();
};

class IncreasePrecisionForQwen3: public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("IncreasePrecisionForQwen3");
    IncreasePrecisionForQwen3();
};

}   // namespace ov::intel_gpu