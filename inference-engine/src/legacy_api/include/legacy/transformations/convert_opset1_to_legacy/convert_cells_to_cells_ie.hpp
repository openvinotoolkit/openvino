// Copyright (C) 2018-2020 Intel Corporation
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

class INFERENCE_ENGINE_API_CLASS(ConvertLSTMCellMatcher);
class INFERENCE_ENGINE_API_CLASS(ConvertGRUCellMatcher);
class INFERENCE_ENGINE_API_CLASS(ConvertRNNCellMatcher);

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertLSTMCellMatcher : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertLSTMCellMatcher();
};

class ngraph::pass::ConvertGRUCellMatcher : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertGRUCellMatcher();
};

class ngraph::pass::ConvertRNNCellMatcher : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertRNNCellMatcher();
};
