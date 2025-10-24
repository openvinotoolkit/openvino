// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/matcher_pass.hpp"

namespace ov::intel_cpu {

class MoE2GeMM : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("MoE2GeMM");
    MoE2GeMM();
};

class MoE3GeMM : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("MoE3GeMM");
    MoE3GeMM();
};

class ConvertMoEMatMuls : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("ConvertMoEMatMuls");
    ConvertMoEMatMuls() {
        add_matcher<ov::intel_cpu::MoE2GeMM>();
        add_matcher<ov::intel_cpu::MoE3GeMM>();
    }
};

}  // namespace ov::intel_cpu
