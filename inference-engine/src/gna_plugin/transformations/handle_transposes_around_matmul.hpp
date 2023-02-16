// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace GNAPluginNS {

struct VerifyReshape {
    bool operator()(const ngraph::Output<ngraph::Node>& reshape_out) const;
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
    HandleTransposesAroundMatMul();
};

} // namespace GNAPluginNS
