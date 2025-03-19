// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_cpu::tpp::pass {

/**
 * @interface EltwiseToEltwiseTPP
 * @brief Converts elementwise operations supported by the TPP backend to the dedicated TPP opset
 * @ingroup snippets
 */
class EltwiseToEltwiseTPP : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("EltwiseToEltwiseTPP");
    EltwiseToEltwiseTPP();
};

}  // namespace ov::intel_cpu::tpp::pass
