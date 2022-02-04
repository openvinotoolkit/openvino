// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <ie_api.h>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class Reshape1DOps;
class Reshape1DConvolution;
class Reshape1DAvgPool;
class Reshape1DMaxPool;

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