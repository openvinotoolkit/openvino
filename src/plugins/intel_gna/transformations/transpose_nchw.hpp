// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace GNAPluginNS {

/**
 * @brief TODO
 */
class TransposeNCHW : public ngraph::pass::MatcherPass {
public:
  NGRAPH_RTTI_DECLARATION;
  TransposeNCHW();
};

} // namespace GNAPluginNS
