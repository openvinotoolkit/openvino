// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <ngraph/ngraph.hpp>

namespace ngraph {
namespace pass {

class DecomposeDivideMatcher;
class ReluReluFusionMatcher;

}  // namespace pass
}  // namespace ngraph

// ! [graph_rewrite:template_transformation_hpp]
// template_pattern_transformation.hpp
class ngraph::pass::DecomposeDivideMatcher: public ngraph::pass::MatcherPass {
public:
    DecomposeDivideMatcher();
};
// ! [graph_rewrite:template_transformation_hpp]

class ngraph::pass::ReluReluFusionMatcher: public ngraph::pass::MatcherPass {
public:
    ReluReluFusionMatcher();
};
