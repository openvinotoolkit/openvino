// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API LSTMCellDecomposition;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
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

class ov::pass::LSTMCellDecomposition : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("LSTMCellDecomposition", "0");
    LSTMCellDecomposition();
};
