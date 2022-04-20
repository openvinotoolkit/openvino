// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace GNAPluginNS {

/**
 * @brief remove concat layers with single input
 *
 * Searches for next pattern
 *     Any input layer
 *           |
 *         Concat
 *           |
 *     Any output layer
 *
 * And transforms to
 *     Any input layer
 *           |
 *     Any output layer
 */
class RemoveSingleInputConcat : public ngraph::pass::MatcherPass {
public:
  OPENVINO_RTTI("RemoveSingleInputConcat", "0");
  RemoveSingleInputConcat();
};

} // namespace GNAPluginNS
