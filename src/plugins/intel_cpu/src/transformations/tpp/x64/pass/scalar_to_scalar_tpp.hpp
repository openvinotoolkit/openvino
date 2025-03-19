// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_cpu::tpp::pass {

/**
 * @interface ScalarToScalarTPP
 * @brief Converts snippets::op::Scalar to tpp::op::Scalar, since TPP operations require a dedicated emitter
 * @ingroup snippets
 */
class ScalarToScalarTPP : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ScalarToScalarTPP");
    ScalarToScalarTPP();
};

}  // namespace ov::intel_cpu::tpp::pass
