// Copyright (C) ß2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/ngraph.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pattern/matcher.hpp>


namespace ngraph {
namespace snippets {
namespace pass {

class Markup : public ngraph::pass::FunctionPass {
public:
    OPENVINO_RTTI("ConvolutionDecomposition", "0");
    bool run_on_model(const std::shared_ptr<ngraph::Function>& m) override;
};

}  // namespace pass
}  // namespace snippets
}  // namespace ngraph
