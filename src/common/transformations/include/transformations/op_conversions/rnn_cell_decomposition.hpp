// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <vector>

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

class ngraph::pass::RNNCellDecomposition : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("RNNCellDecomposition", "0");
    RNNCellDecomposition();
};
