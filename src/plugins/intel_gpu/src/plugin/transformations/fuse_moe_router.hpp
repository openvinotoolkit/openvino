// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

class FuseMoESoftmaxRouter : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseMoESoftmaxRouter");
    FuseMoESoftmaxRouter();
};

class FuseMoESigmoidRouter : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseMoESigmoidRouter");
    FuseMoESigmoidRouter();
};

/// GraphRewrite that applies Mixture of Experts routing subgraph fusion
class FuseMoERouter : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("FuseMoERouter");
    FuseMoERouter();
};

}  // namespace ov::intel_gpu
