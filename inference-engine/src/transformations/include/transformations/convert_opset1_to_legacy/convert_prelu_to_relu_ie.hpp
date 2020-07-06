// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>
#include <string>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ConvertPReLUToReLUIEMatcher;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertPReLUToReLUIEMatcher: public ngraph::pass::MatcherPass {
public:
    ConvertPReLUToReLUIEMatcher();
};
