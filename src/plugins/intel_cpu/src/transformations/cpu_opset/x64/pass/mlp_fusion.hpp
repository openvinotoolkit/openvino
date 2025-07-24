// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pass.hpp"

namespace ov::intel_cpu {

class MLPFusionPass : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("MLPFusionPass");
    MLPFusionPass();
};

class MLPFusion : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("MLPFusion");
    MLPFusion() = default;
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::intel_cpu
