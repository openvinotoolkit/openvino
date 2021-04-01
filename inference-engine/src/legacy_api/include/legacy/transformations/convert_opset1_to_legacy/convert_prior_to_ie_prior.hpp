// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <ie_api.h>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class INFERENCE_ENGINE_API_CLASS(ConvertPriorBox);
class INFERENCE_ENGINE_API_CLASS(ConvertPriorBoxToLegacy);
class INFERENCE_ENGINE_API_CLASS(ConvertPriorBoxClusteredToLegacy);

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertPriorBoxToLegacy : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertPriorBoxToLegacy();
};

class ngraph::pass::ConvertPriorBoxClusteredToLegacy : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertPriorBoxClusteredToLegacy();
};

class ngraph::pass::ConvertPriorBox: public ngraph::pass::GraphRewrite {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertPriorBox() {
        add_matcher<ngraph::pass::ConvertPriorBoxToLegacy>();
        add_matcher<ngraph::pass::ConvertPriorBoxClusteredToLegacy>();
    }
};