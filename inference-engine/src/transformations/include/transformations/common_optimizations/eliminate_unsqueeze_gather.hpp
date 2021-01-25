// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>

#include <ngraph/ngraph.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include "ngraph/pattern/matcher.hpp"

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API EliminateUnsqueezeGather;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief Remove Unsqueeze + Gather pair, if Gather gathers data by dimension
 * that was previously added by Unsqueeze
 */

class ngraph::pass::EliminateUnsqueezeGather : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    EliminateUnsqueezeGather();
};
