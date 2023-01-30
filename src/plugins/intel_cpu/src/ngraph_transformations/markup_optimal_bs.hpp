// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_cpu {
namespace mixed_affinity {

class MarkupConvolutionOptimalBS: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    MarkupConvolutionOptimalBS();
};

class MarkupGroupConvolutionOptimalBS : public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    MarkupGroupConvolutionOptimalBS();
};

class MarkupFullyConnectedOptimalBS : public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    MarkupFullyConnectedOptimalBS();
};

class MarkupOptimalBS : public ov::pass::GraphRewrite {
public:
    NGRAPH_RTTI_DECLARATION;
    MarkupOptimalBS();
};

}  // namespace mixed_affinity
}  // namespace intel_cpu
}  // namespace ov
