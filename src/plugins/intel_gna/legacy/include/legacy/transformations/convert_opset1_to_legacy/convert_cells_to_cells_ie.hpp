// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_api.h>

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <string>
#include <vector>

namespace ngraph {
namespace pass {

class ConvertLSTMCellMatcher;
class ConvertGRUCellMatcher;
class ConvertRNNCellMatcher;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertLSTMCellMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertLSTMCellMatcher", "0");
    ConvertLSTMCellMatcher();
};

class ngraph::pass::ConvertGRUCellMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertGRUCellMatcher", "0");
    ConvertGRUCellMatcher();
};

class ngraph::pass::ConvertRNNCellMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertRNNCellMatcher", "0");
    ConvertRNNCellMatcher();
};
