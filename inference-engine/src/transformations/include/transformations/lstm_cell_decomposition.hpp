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

class TRANSFORMATIONS_API LSTMCellDecomposition;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief LSTMCellDecomposition transformation decomposes LSTMCell layer with inputs X, H, C, W, R, B
 * to Add, Split, MatMul, Multiply ops according to the formula:
 *              (.) - Denotes element-wise multiplication.
                *   - Denotes dot product.
                f, g, h - are activation functions.

 *              it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
                ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Wbf + Rbf)
                ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
                ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Wbo + Rbo)
                Ct = ft (.) Ct-1 + it (.) ct
                Ht = ot (.) h(Ct)
 * *
 */

class ngraph::pass::LSTMCellDecomposition: public ngraph::pass::MatcherPass {
public:
    LSTMCellDecomposition();
};
