// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace GNAPluginNS {

struct VerifyReshape {
    bool operator()(const ngraph::Output<ngraph::Node>& reshape_out) const {
        auto in_shape = reshape_out.get_node_shared_ptr()->get_input_shape(0);
        auto out_shape = reshape_out.get_node_shared_ptr()->get_output_shape(0);

        // Check if Reshape changes the final 2d shape of Affine primitive
        in_shape.erase(std::remove(in_shape.begin(), in_shape.end(), 1), in_shape.end());
        out_shape.erase(std::remove(out_shape.begin(), out_shape.end(), 1), out_shape.end());
        return in_shape != out_shape;
    }
};

/**
 * @brief Inserts Transpose before MatMul or removes it (if it exists) if there is Reshape
 * before MatMul which changes the batch size:
 *    [1, A*B]                 [1, A*B]
 *       |                       |
 *    Reshape                 Reshape
 *       |                       |
 * [1, A, 1, B]            [1, A, 1, B]
 *       |                       |
 *       |                   Transpose
 *       |           ->          |
 *       |           <-     [1, B, 1, A]
 *       |                       |
 *    MatMul                   MatMul
 */
class HandleTransposeBeforeMatMul : public ngraph::pass::MatcherPass {
public:
  NGRAPH_RTTI_DECLARATION;
  HandleTransposeBeforeMatMul();
};

/**
 * @brief Inserts Transpose after MatMul or removes it (if it exists) if there is Reshape
 * after MatMul which changes the batch size:
 *    MatMul                  MatMul
 *       |                       |
 * [1, A, 1, B]            [1, A, 1, B]
 *       |                       |
 *       |                   Transpose
 *       |           ->          |
 *       |           <-     [1, B, 1, A]
 *       |                       |
 *    Reshape                 Reshape
 *       |                       |
 *    [1, A*B]                [1, A*B]
 */
class HandleTransposeAfterMatMul : public ngraph::pass::MatcherPass {
public:
  NGRAPH_RTTI_DECLARATION;
  HandleTransposeAfterMatMul();
};

class HandleTransposesAroundMatMul: public ngraph::pass::GraphRewrite {
public:
    NGRAPH_RTTI_DECLARATION;
    HandleTransposesAroundMatMul() {
        add_matcher<HandleTransposeBeforeMatMul>();
        add_matcher<HandleTransposeAfterMatMul>();
    }
};

} // namespace GNAPluginNS