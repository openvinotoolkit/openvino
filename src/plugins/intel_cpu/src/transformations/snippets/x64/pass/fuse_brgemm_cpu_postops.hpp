// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_cpu::pass {

class FuseScaleShift : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseScaleShift");
    FuseScaleShift();
};

class FuseBinaryEltwise : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseBinaryEltwise");
    FuseBinaryEltwise();

private:
    size_t m_fused_postops_count = 0;
};

class FuseBrgemmCPUPostops : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("FuseBrgemmCPUPostops");
    FuseBrgemmCPUPostops() {
        add_matcher<FuseScaleShift>();
        add_matcher<FuseBinaryEltwise>();
    }
};

}  // namespace ov::intel_cpu::pass
