// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ShuffleChannelsFusion;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief ShuffleChannelsFusion transformation replaces following graph:
 * Reshape->Transpose->Reshape to ShuffleChannels
 * Restrictions:
 * - transpose permutation has to be {0, 2, 1, 3, 4}
 * - first reshape input shape has to be equal the second reshape output shape
 */

class ngraph::pass::ShuffleChannelsFusion: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ShuffleChannelsFusion();
};
