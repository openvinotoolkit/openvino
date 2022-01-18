// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace GNAPluginNS {

/**
 * @brief Inserts Transpose before MatMul or removes it (if it exists) if there is Reshape
 * before MatMul which changes the batch size:
 *    [1, A*B]                [1, A*B]
 *       |                       |
 *    Reshape                 Reshape
 *       |                       |
 *    [A, B]                  [A, B]
 *       |                       |
 *       |                   Transpose
 *       |           ->          |
 *       |           <-       [B, A]
 *       |                       |
 *       |                    Reshape
 *       |                    [A, B]
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
class HandleTransposeAfterMatMul: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    HandleTransposeAfterMatMul();
};

class HandleTransposesAroundMatMul : public ngraph::pass::GraphRewrite {
public:
  NGRAPH_RTTI_DECLARATION;
  HandleTransposesAroundMatMul();
};

} // namespace GNAPluginNS
