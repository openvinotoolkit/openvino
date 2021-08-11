// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>
#include <string>

#include <ie_api.h>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace pass {

class INFERENCE_ENGINE_API_CLASS(ConvertLSTMCellMatcher);
class INFERENCE_ENGINE_API_CLASS(ConvertGRUCellMatcher);
class INFERENCE_ENGINE_API_CLASS(ConvertRNNCellMatcher);

}  // namespace pass
}  // namespace ov

class ov::pass::ConvertLSTMCellMatcher : public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertLSTMCellMatcher();
};

class ov::pass::ConvertGRUCellMatcher : public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertGRUCellMatcher();
};

class ov::pass::ConvertRNNCellMatcher : public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertRNNCellMatcher();
};
