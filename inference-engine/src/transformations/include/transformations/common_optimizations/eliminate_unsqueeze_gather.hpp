// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>

#include <ngraph/ngraph.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include "ngraph/pattern/matcher.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API EliminateUnsqueezeGather;
class TRANSFORMATIONS_API EliminateGatherUnsqueeze;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief Remove Unsqueeze + Gather pair, if Gather gathers data by dimension
 * that was previously added by Unsqueeze
 */

class ov::pass::EliminateUnsqueezeGather : public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    EliminateUnsqueezeGather();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief Remove Gather -> Unsqueeze pair, if Gather takes a scalar and
 * Unsqueeze makes it a 1D tensor
 */

class ov::pass::EliminateGatherUnsqueeze : public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    EliminateGatherUnsqueeze();
};
