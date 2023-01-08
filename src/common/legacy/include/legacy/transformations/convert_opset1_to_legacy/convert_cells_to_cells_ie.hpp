// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>
#include <string>

#include <ie_api.h>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class ConvertLSTMCellMatcher;
class ConvertGRUCellMatcher;
class ConvertRNNCellMatcher;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertLSTMCellMatcher : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertLSTMCellMatcher", "0");
    ConvertLSTMCellMatcher();
};

class ngraph::pass::ConvertGRUCellMatcher : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertGRUCellMatcher", "0");
    ConvertGRUCellMatcher();
};

class ngraph::pass::ConvertRNNCellMatcher : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertRNNCellMatcher", "0");
    ConvertRNNCellMatcher();
};
