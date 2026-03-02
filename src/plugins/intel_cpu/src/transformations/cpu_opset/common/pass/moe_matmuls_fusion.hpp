// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/matcher_pass.hpp"

namespace ov::intel_cpu {

class MoE2GeMMFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("MoE2GeMMFusion");
    MoE2GeMMFusion();
};

class MoE3GeMMFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("MoE3GeMMFusion");
    MoE3GeMMFusion();
};

class MoEMatMulsFusion : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("MoEMatMulsFusion");
    MoEMatMulsFusion() {
        add_matcher<ov::intel_cpu::MoE2GeMMFusion>();
        add_matcher<ov::intel_cpu::MoE3GeMMFusion>();
    }
};

}  // namespace ov::intel_cpu
