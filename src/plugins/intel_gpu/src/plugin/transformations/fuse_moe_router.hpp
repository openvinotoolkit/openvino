// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

/// Fuse softmax routing subgraph (MatMul → Softmax → TopK → Normalize → Transpose → Unsqueeze)
/// into MoERouterFused, connecting its outputs directly to MOECompressed inputs.
class FuseMoESoftmaxRouter : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseMoESoftmaxRouter");
    FuseMoESoftmaxRouter();
};

/// Fuse sigmoid+bias routing subgraph (MatMul → Sigmoid → Add → TopK → Normalize → Transpose → Unsqueeze)
/// into MoERouterFused, connecting its outputs directly to MOECompressed inputs.
class FuseMoESigmoidRouter : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseMoESigmoidRouter");
    FuseMoESigmoidRouter();
};

/// GraphRewrite that applies both softmax and sigmoid routing fusion passes.
class FuseMoERouter : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("FuseMoERouter");
    FuseMoERouter();
};

}  // namespace ov::intel_gpu
