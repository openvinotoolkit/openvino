// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>


namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ConvertNormalizeL2WithMulToNormalizeIE;
class TRANSFORMATIONS_API ConvertNormalizeL2ToLegacyMatcher;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertNormalizeL2WithMulToNormalizeIE: public ngraph::pass::MatcherPass {
public:
    ConvertNormalizeL2WithMulToNormalizeIE();
};

class ngraph::pass::ConvertNormalizeL2ToLegacyMatcher: public ngraph::pass::MatcherPass {
public:
    ConvertNormalizeL2ToLegacyMatcher();
};
