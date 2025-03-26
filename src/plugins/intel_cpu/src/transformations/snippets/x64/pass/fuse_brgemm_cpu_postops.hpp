// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_cpu::pass {

/**
 * @interface FuseBrgemmCPUPostopsLegacy
 * @brief TODO
 * @ingroup snippets
 */
class FuseBrgemmCPUPostopsLegacy : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseBrgemmCPUPostopsLegacy");
    FuseBrgemmCPUPostopsLegacy();
};

class FuseBrgemmCPUPostops : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("FuseBrgemmCPUPostops");
    FuseBrgemmCPUPostops() {
        if (std::getenv("LEGACY")) {
            add_matcher<FuseBrgemmCPUPostopsLegacy>();
        }
    }
};

}  // namespace ov::intel_cpu::pass
