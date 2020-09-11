// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ConvertConvolutions;

class TRANSFORMATIONS_API ConvertConvolution;
class TRANSFORMATIONS_API ConvertGroupConvolution;
class TRANSFORMATIONS_API ConvertDeconvolution;
class TRANSFORMATIONS_API ConvertGroupDeconvolution;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertConvolutions: public ngraph::pass::GraphRewrite {
public:
    ConvertConvolutions() {
        add_matcher<ngraph::pass::ConvertConvolution>();
        add_matcher<ngraph::pass::ConvertGroupConvolution>();
        add_matcher<ngraph::pass::ConvertDeconvolution>();
        add_matcher<ngraph::pass::ConvertGroupDeconvolution>();
    }
};

class ngraph::pass::ConvertConvolution: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertConvolution();
};

class ngraph::pass::ConvertGroupConvolution: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertGroupConvolution();
};

class ngraph::pass::ConvertDeconvolution: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertDeconvolution();
};

class ngraph::pass::ConvertGroupDeconvolution: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertGroupDeconvolution();
};