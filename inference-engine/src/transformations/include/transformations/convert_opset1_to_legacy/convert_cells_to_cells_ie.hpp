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

class TRANSFORMATIONS_API ConvertLSTMCellMatcher;
class TRANSFORMATIONS_API ConvertGRUCellMatcher;
class TRANSFORMATIONS_API ConvertRNNCellMatcher;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertLSTMCellMatcher : public ngraph::pass::MatcherPass {
public:
    ConvertLSTMCellMatcher();
};

class ngraph::pass::ConvertGRUCellMatcher : public ngraph::pass::MatcherPass {
public:
    ConvertGRUCellMatcher();
};

class ngraph::pass::ConvertRNNCellMatcher : public ngraph::pass::MatcherPass {
public:
    ConvertRNNCellMatcher();
};
