// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API EinsumDecomposition;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief EinsumDecomposition transformation decomposes Einsum-7 operation into a sub-graph with more simple operations:
 *        Transpose, Reshape, MatMul, ReduceSum, Unsqueeze, ShapeOf, ReduceProd, StridedSlice, and Concat
 */
class ov::pass::EinsumDecomposition : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("EinsumDecomposition", "0");
    EinsumDecomposition();
};
