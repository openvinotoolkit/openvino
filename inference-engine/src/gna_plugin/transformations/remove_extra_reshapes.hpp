// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace GNAPluginNS {

/**
 * @brief Removes reshapes before MaxPool which do nothing. Such reshapes can be a result of conversion from IR10 to IR7.
 */
class RemoveExtraReshapes : public ngraph::pass::MatcherPass {
public:
  NGRAPH_RTTI_DECLARATION;
  RemoveExtraReshapes();
};

} // namespace GNAPluginNS