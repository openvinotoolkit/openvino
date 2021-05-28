// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace GNAPluginNS {

/**
 * @brief Inserts Transpose before MatMul in the following topology:
 *       [1, A]
 *          |
 *       Reshape
 *          |
 *       [B, C],
 *    1 < B <= 8, C % 8 == 0 or
 *    B % 8 == 0, 1 < C <= 8
 *         |             Const
 *          \             /
 *               Matmul
 */
class InsertTransposeBeforeMatmul : public ngraph::pass::MatcherPass {
public:
  NGRAPH_RTTI_DECLARATION;
  InsertTransposeBeforeMatmul();
};

} // namespace GNAPluginNS