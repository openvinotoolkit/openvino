// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API RNNCellDecomposition;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief RNNCellDecomposition transformation decomposes RNNCell layer with inputs X, H, W, R, B
 * to Add, MatMul ops according to the formula:
                *   - Denotes dot product.
                f - is an activation functions.

 *              Ht = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
 * *
 */

class ov::pass::RNNCellDecomposition : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("RNNCellDecomposition");
    RNNCellDecomposition();
};
