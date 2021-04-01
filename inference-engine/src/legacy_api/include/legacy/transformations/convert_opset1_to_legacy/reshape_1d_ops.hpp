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

class INFERENCE_ENGINE_API_CLASS(Reshape1DOps);
class INFERENCE_ENGINE_API_CLASS(Reshape1DConvolution);
class INFERENCE_ENGINE_API_CLASS(Reshape1DAvgPool);
class INFERENCE_ENGINE_API_CLASS(Reshape1DMaxPool);

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::Reshape1DConvolution: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    Reshape1DConvolution();
};

class ngraph::pass::Reshape1DAvgPool: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    Reshape1DAvgPool();
};

class ngraph::pass::Reshape1DMaxPool: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    Reshape1DMaxPool();
};

class ngraph::pass::Reshape1DOps: public ngraph::pass::GraphRewrite {
public:
    NGRAPH_RTTI_DECLARATION;
    Reshape1DOps() {
        add_matcher<ngraph::pass::Reshape1DConvolution>();
        add_matcher<ngraph::pass::Reshape1DAvgPool>();
        add_matcher<ngraph::pass::Reshape1DMaxPool>();
    }
};