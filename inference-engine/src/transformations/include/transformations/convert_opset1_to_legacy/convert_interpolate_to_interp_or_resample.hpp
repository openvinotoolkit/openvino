// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <set>
#include <string>
#include <vector>
#include <memory>
#include <map>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ConvertInterpolateToInterpOrResampleMatcher;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertInterpolateToInterpOrResampleMatcher: public ngraph::pass::MatcherPass {
public:
    ConvertInterpolateToInterpOrResampleMatcher();
};
