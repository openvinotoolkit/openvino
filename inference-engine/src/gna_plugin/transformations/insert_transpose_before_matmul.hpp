// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace GNAPluginNS {

/**
 * @brief Insers Transpose before MatMul in the following topology:
 */
class InsertTransposeBeforeMatmul : public ngraph::pass::MatcherPass {
public:
  NGRAPH_RTTI_DECLARATION;
  InsertTransposeBeforeMatmul();
};

} // namespace GNAPluginNS