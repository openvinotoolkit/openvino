// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API EinsumDecomposition;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief EinsumDecomposition transformation decomposes Einsum-7 operation into a sub-graph with more simple operations:
 *        Transpose, Reshape, MatMul, ReduceSum, Unsqueeze, ShapeOf, ReduceProd, StridedSlice, and Concat
 */
class ov::pass::EinsumDecomposition : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("EinsumDecomposition");
    EinsumDecomposition();
};
