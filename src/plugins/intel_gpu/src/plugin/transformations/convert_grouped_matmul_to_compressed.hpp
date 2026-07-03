// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/matcher_pass.hpp"

namespace ov::intel_gpu {

// Matches the 2D x 3D form: GroupedMatMul(data, compressed_weights, offsets).
class ConvertGroupedMatMulWithOffsetsToCompressed : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertGroupedMatMulWithOffsetsToCompressed");
    ConvertGroupedMatMulWithOffsetsToCompressed();
};

// Matches the 3D x 3D form: GroupedMatMul(data, compressed_weights) (no offsets).
class ConvertGroupedMatMulNoOffsetsToCompressed : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertGroupedMatMulNoOffsetsToCompressed");
    ConvertGroupedMatMulNoOffsetsToCompressed();
};

// Composite pass that runs both matchers above. GraphRewrite is required because
// each MatcherPass can only hold a single matcher, but v17::GroupedMatMul has two
// legal input arities (2 and 3) that need separate root patterns.
class ConvertGroupedMatMulToGroupedMatMulCompressed : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("ConvertGroupedMatMulToGroupedMatMulCompressed");
    ConvertGroupedMatMulToGroupedMatMulCompressed();
};

}  // namespace ov::intel_gpu
