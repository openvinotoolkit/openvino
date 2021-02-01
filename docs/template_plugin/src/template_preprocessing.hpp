// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include "ie_input_info.hpp"

namespace ngraph {
namespace pass {

class AddPreprocessingMatcher;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::AddPreprocessingMatcher : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    AddPreprocessingMatcher(const InferenceEngine::InputsDataMap & inputInfoMap);
};
