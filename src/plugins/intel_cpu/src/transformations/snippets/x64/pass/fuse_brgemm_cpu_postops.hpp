// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_cpu::pass {

/**
 * @interface FuseBrgemmCPUPostops
 * @brief TODO
 * @ingroup snippets
 */
class FuseBrgemmCPUPostops : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseBrgemmCPUPostops");
    FuseBrgemmCPUPostops();
};

class FuseBrgemmOutConvert : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseBrgemmOutConvert");
    FuseBrgemmOutConvert();
};

}  // namespace ov::intel_cpu::pass
