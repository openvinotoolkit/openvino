// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/ngraph.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

#include "ngraph/pattern/matcher.hpp"

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API EliminateUnsqueezeGather;
class TRANSFORMATIONS_API EliminateGatherUnsqueeze;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief Remove Unsqueeze + Gather pair, if Gather gathers data by dimension
 * that was previously added by Unsqueeze
 */

class ngraph::pass::EliminateUnsqueezeGather : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("EliminateUnsqueezeGather", "0");
    EliminateUnsqueezeGather();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief Remove Gather -> Unsqueeze pair, if Gather takes a scalar and
 * Unsqueeze makes it a 1D tensor
 */

class ngraph::pass::EliminateGatherUnsqueeze : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("EliminateGatherUnsqueeze", "0");
    EliminateGatherUnsqueeze();
};
