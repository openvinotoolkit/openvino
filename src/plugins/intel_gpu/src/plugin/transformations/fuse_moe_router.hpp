// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "openvino/core/type/element_type.hpp"
#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

class FuseMoESoftmaxRouter : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseMoESoftmaxRouter");
    explicit FuseMoESoftmaxRouter(const std::vector<ov::element::Type>& supported_activation_types);
};

class FuseMoESigmoidRouter : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseMoESigmoidRouter");
    explicit FuseMoESigmoidRouter(const std::vector<ov::element::Type>& supported_activation_types);
};

/// GraphRewrite that applies Mixture of Experts routing subgraph fusion
class FuseMoERouter : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("FuseMoERouter");
    explicit FuseMoERouter(const std::vector<ov::element::Type>& supported_activation_types);
};

}  // namespace ov::intel_gpu
