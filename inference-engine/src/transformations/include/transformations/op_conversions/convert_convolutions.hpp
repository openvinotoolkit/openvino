// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertConvolutions;

class TRANSFORMATIONS_API ConvertConvolution;
class TRANSFORMATIONS_API ConvertGroupConvolution;
class TRANSFORMATIONS_API ConvertDeconvolution;
class TRANSFORMATIONS_API ConvertGroupDeconvolution;

}  // namespace pass
}  // namespace ov

class ov::pass::ConvertConvolution: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertConvolution();
};

class ov::pass::ConvertGroupConvolution: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertGroupConvolution();
};

class ov::pass::ConvertDeconvolution: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertDeconvolution();
};

class ov::pass::ConvertGroupDeconvolution: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertGroupDeconvolution();
};

class ov::pass::ConvertConvolutions: public ov::pass::GraphRewrite {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertConvolutions() {
        add_matcher<ov::pass::ConvertConvolution>();
        add_matcher<ov::pass::ConvertGroupConvolution>();
        add_matcher<ov::pass::ConvertDeconvolution>();
        add_matcher<ov::pass::ConvertGroupDeconvolution>();
    }
};
