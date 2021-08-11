// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <ie_api.h>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace pass {

class INFERENCE_ENGINE_API_CLASS(Reshape1DOps);
class INFERENCE_ENGINE_API_CLASS(Reshape1DConvolution);
class INFERENCE_ENGINE_API_CLASS(Reshape1DAvgPool);
class INFERENCE_ENGINE_API_CLASS(Reshape1DMaxPool);

}  // namespace pass
}  // namespace ov

class ov::pass::Reshape1DConvolution: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    Reshape1DConvolution();
};

class ov::pass::Reshape1DAvgPool: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    Reshape1DAvgPool();
};

class ov::pass::Reshape1DMaxPool: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    Reshape1DMaxPool();
};

class ov::pass::Reshape1DOps: public ov::pass::GraphRewrite {
public:
    NGRAPH_RTTI_DECLARATION;
    Reshape1DOps() {
        add_matcher<ov::pass::Reshape1DConvolution>();
        add_matcher<ov::pass::Reshape1DAvgPool>();
        add_matcher<ov::pass::Reshape1DMaxPool>();
    }
};
