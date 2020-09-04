// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API GRUCellDecomposition;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief GRUCellDecomposition transformation decomposes GRUCell layer with inputs X, H, W, R, B
 * to Add, Split, MatMul, Multiply and Subtract ops according to the formula:
                (.) - Denotes element-wise multiplication.
                *   - Denotes dot product.
                f, g  - are activation functions

                zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
                rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
                ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh) # when linear_before_reset := false # (default)
                ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh) # when linear_before_reset:= true
                Ht = (1 - zt) (.) ht + zt (.) Ht-1
 * *
 */

class ngraph::pass::GRUCellDecomposition: public ngraph::pass::MatcherPass {
public:
    GRUCellDecomposition();
};
