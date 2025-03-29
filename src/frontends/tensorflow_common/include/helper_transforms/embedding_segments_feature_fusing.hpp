// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {
namespace pass {

// The transformation looks for pattern (sub-graph) that performs extraction of embedding vectors from the parameters
// table for object feature values, and sum up these embedding vectors for every object or compute their mean value.
// Such sub-graph is met in the Wide and Deep model in case of the SINGLE categorical feature.
class EmbeddingSegmentSingleFeatureFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::tensorflow::pass::EmbeddingSegmentSingleFeatureFusion");
    EmbeddingSegmentSingleFeatureFusion();
};

}  // namespace pass
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
