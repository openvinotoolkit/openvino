// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace GNAPluginNS {

/**
 * @brief Brodcast data in Const layer
 * Transformation recognizes the next patterns
 *
 * Constant    Any
 *       |     |
 *       Eltwise
 *
 * Constant
 *    |
 * FakeQuantize     Any
 *            |     |
 *            Eltwise
 *
 * Where Eltwise node is one of the: Multiply, Substract and Add
 *
 * If eltwise node inputs have different shapes and one the inputs is Constant node
 * we can update (broadcast) Constant to have the same shape as another input.
 * Broadcasting may be done if shapes are compartible - suitable for that.
 */
class BroadcastAddMultiplyConst : public ngraph::pass::MatcherPass {
public:
  NGRAPH_RTTI_DECLARATION;
  BroadcastAddMultiplyConst();
};

} // namespace GNAPluginNS
