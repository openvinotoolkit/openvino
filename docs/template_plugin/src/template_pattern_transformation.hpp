// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class DecomposeDivideMatcher;
class ReluReluFusionMatcher;

}  // namespace pass
}  // namespace ngraph

// ! [graph_rewrite:template_transformation_hpp]
// template_pattern_transformation.hpp
/**
 * @ingroup ie_transformation_common_api
 * @brief Add transformation description.
 */
class ngraph::pass::DecomposeDivideMatcher: public ngraph::pass::MatcherPass {
public:
    DecomposeDivideMatcher();
};
// ! [graph_rewrite:template_transformation_hpp]

class ngraph::pass::ReluReluFusionMatcher: public ngraph::pass::MatcherPass {
public:
    ReluReluFusionMatcher();
};
