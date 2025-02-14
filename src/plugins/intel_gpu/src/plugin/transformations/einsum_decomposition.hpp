// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/core/visibility.hpp"

namespace ov::intel_gpu {

/**
 * @brief EinsumDecomposition transformation decomposes Einsum-7 operation into a sub-graph with more simple operations:
 *        Transpose, Reshape, MatMul, ReduceSum, Unsqueeze, ShapeOf, ReduceProd, StridedSlice, and Concat
 */
class EinsumDecomposition : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("EinsumDecomposition");
    EinsumDecomposition();
};

}  // namespace ov::intel_gpu
