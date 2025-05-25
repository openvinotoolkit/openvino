// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_cpu::pass {

/**
 * @interface BrgemmToGemmCPU
 * @brief The pass decompose Snippets Brgemm to specific ops
 *        Brgemm -> copyB + GemmCPU
 * @ingroup snippets
 */
class BrgemmToGemmCPU : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("BrgemmToGemmCPU");
    BrgemmToGemmCPU();
};

}  // namespace ov::intel_cpu::pass
