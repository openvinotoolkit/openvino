// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <sstream>
#include <vector>

#include <ie_api.h>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {
    class INFERENCE_ENGINE_API_CLASS(HandleGemmLayer);
    class INFERENCE_ENGINE_API_CLASS(HandleGemmLayerPass);
} // namespace pass
} // namespace ngraph

class ngraph::pass::HandleGemmLayerPass: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    HandleGemmLayerPass();
};

class ngraph::pass::HandleGemmLayer: public ngraph::pass::GraphRewrite {
public:
    NGRAPH_RTTI_DECLARATION;
    HandleGemmLayer() {
        add_matcher<ngraph::pass::HandleGemmLayerPass>();
    }
};