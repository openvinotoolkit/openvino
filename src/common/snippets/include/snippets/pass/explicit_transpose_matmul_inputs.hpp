// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov {
namespace snippets {
namespace pass {

/**
 * @interface ExplicitTransposeMatMulInputs
 * @brief The pass extracts explicit Transpose node from MatMul with transposed_<a|b> and moves it to Parameter.
 *        If there is another Transpose, the pass fuses extracted Transpose and existing Transpose.
 *        For example, At the moment Snippets supports Transpose only with order {0, 2, 3, 1}, so if there is pattern in graph:
 *                  in0     Transpose{0, 2, 1, 3}
 *                    \    /
 *                    MatMul[false, true]
 *        We can set `false` in MatMul parameter `transposed_b` and change Transpose order to {0, 2, 3, 1} which is supported by Snippets
 * @ingroup snippets
 */
class ExplicitTransposeMatMulInputs: public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ExplicitTransposeMatMulInputs", "0");
    ExplicitTransposeMatMulInputs();

    // Return `True` if all inputs (except 0-th input) have scalar shape. Otherwise returns `False`
    static bool are_weights_scalar(const std::shared_ptr<ov::Node>& node);

private:
    static void extract(const ov::Input<ov::Node>& input);
};

}  // namespace pass
}  // namespace snippets
}  // namespace ov
