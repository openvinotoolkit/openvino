// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/ngraph.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pattern/matcher.hpp>


namespace ngraph {
namespace snippets {
namespace pass {

class ConvolutionDecomposition : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvolutionDecomposition", "0");
    explicit ConvolutionDecomposition();
};

}  // namespace pass
}  // namespace snippets
}  // namespace ngraph
