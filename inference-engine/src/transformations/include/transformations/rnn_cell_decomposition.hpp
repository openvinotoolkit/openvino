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

class TRANSFORMATIONS_API RNNCellDecomposition;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief RNNCellDecomposition transformation decomposes RNNCell layer with inputs X, H, W, R, B
 * to Add, MatMul ops according to the formula:
                *   - Denotes dot product.
                f - is an activation functions.

 *              Ht = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
 * *
 */

class ngraph::pass::RNNCellDecomposition: public ngraph::pass::MatcherPass {
public:
    RNNCellDecomposition();
};
