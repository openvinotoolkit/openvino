// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ov {
namespace runtime {
namespace intel_gpu {

/**
 * @brief EinsumDecomposition transformation decomposes Einsum-7 operation into a sub-graph with more simple operations:
 *        Transpose, Reshape, MatMul, ReduceSum, Unsqueeze, ShapeOf, ReduceProd, StridedSlice, and Concat
 */
class EinsumDecomposition : public ngraph::pass::MatcherPass {
public:
    EinsumDecomposition();
};

}  // namespace intel_gpu
}  // namespace runtime
}  // namespace ov
