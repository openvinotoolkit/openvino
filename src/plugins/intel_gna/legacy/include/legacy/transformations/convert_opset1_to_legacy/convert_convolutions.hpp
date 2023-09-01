// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_api.h>

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <vector>

namespace ngraph {
namespace pass {

class ConvertConvolutions;

class ConvertConvolution;
class ConvertGroupConvolution;
class ConvertDeconvolution;
class ConvertGroupDeconvolution;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertConvolution : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertConvolution", "0");
    ConvertConvolution();
};

class ngraph::pass::ConvertGroupConvolution : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertGroupConvolution", "0");
    ConvertGroupConvolution();
};

class ngraph::pass::ConvertDeconvolution : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertDeconvolution", "0");
    ConvertDeconvolution();
};

class ngraph::pass::ConvertGroupDeconvolution : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertGroupDeconvolution", "0");
    ConvertGroupDeconvolution();
};

class ngraph::pass::ConvertConvolutions : public ov::pass::GraphRewrite {
public:
    OPENVINO_RTTI("ConvertConvolutions", "0");
    ConvertConvolutions() {
        add_matcher<ngraph::pass::ConvertConvolution>();
        add_matcher<ngraph::pass::ConvertGroupConvolution>();
        add_matcher<ngraph::pass::ConvertDeconvolution>();
        add_matcher<ngraph::pass::ConvertGroupDeconvolution>();
    }
};
