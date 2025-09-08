// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov::intel_cpu {

class QKVProjFusionPass1 : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("QKVProjFusionPass1");
    QKVProjFusionPass1();
};

class QKVProjFusionPass2 : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("QKVProjFusionPass1");
    QKVProjFusionPass2();
};

class QKVProjFusion : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("QKVProjFusion");
    QKVProjFusion() = default;
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::intel_cpu
