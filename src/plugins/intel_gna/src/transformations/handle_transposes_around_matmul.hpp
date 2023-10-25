// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_gna {
namespace pass {

/**
 * @brief Inserts Transpose before MatMul or removes it (if it exists)
 * if there is Reshape/Concat layer before MatMul which changes the batch size,
 * or transpose the input if there's a Constant/FQ layer on the first input:
 *      [1, A*B]                [1, A*B]
 *         |                       |
 *  Reshape / Concat        Reshape / Concat
 *         |                       |
 *      [A, B]                  [A, B]
 *         |                       |
 *         |                   Transpose
 *         |           ->          |
 *         |           <-       [B, A]
 *         |                       |
 *         |                    Reshape
 *         |                    [A, B]
 *         |                       |
 *      MatMul                   MatMul
 */
class HandleTransposeBeforeMatMul : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("HandleTransposeBeforeMatMul", "0");
    HandleTransposeBeforeMatMul();
};

/**
 * @brief Inserts Transpose after MatMul or removes it (if it exists) if there is Reshape
 * after MatMul which changes the batch size:
 *    MatMul                  MatMul
 *    [A, B]                  [A, B]
 *       |                       |
 *     [Add]                   [Add]
 *       |                       |
 *  [FakeQuantize]        [FakeQuantize]
 *       |                       |
 *   [Activation]          [Activation]
 *       |                       |
 *       |                    Reshape
 *       |                    [B, A]
 *       |                       |
 *       |                   Transpose
 *       |           ->          |
 *       |           <-        [A, B]
 *       |                       |
 *    Reshape                 Reshape
 *       |                       |
 *    [1, A*B]                [1, A*B]
 */
class HandleTransposeAfterMatMul : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("HandleTransposeAfterMatMul", "0");
    HandleTransposeAfterMatMul();
};

class HandleTransposesAroundMatMul : public ov::pass::GraphRewrite {
public:
    OPENVINO_RTTI("HandleTransposesAroundMatMul", "0");
    HandleTransposesAroundMatMul();
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
