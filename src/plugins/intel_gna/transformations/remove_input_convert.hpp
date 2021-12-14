// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace GNAPluginNS {

/**
 * @brief remove convert layers after inputs
 *
 * Searches for next pattern
 *     Any input layer
 *           |
 *         Convert
 *           |
 *        Any layer
 *
 * And transforms to
 *     Any input layer
 *           |
 *        Any layer
 */
class RemoveInputConvert : public ngraph::pass::MatcherPass {
public:
  NGRAPH_RTTI_DECLARATION;
  RemoveInputConvert();
};

/**
 * @brief remove convert layers after before outputs
 *
 * Searches for next pattern
 *        Any layer
 *           |
 *         Convert
 *           |
 *     Any output layer
 *
 * And transforms to
 *        Any layer
 *           |
 *    Any output layer
 */
class RemoveOutputConvert : public ngraph::pass::MatcherPass {
public:
  NGRAPH_RTTI_DECLARATION;
  RemoveOutputConvert();
};

} // namespace GNAPluginNS
