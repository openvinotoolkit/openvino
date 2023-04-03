// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_cpu {

class FullyConnectedBiasFusionWithoutMultiply : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("FullyConnectedBiasFusionWithoutMultiply", "0");
    FullyConnectedBiasFusionWithoutMultiply();
};

class FullyConnectedDQBiasFusion : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("FullyConnectedDQBiasFusion", "0");
    FullyConnectedDQBiasFusion();
};

class FullyConnectedBiasFusion : public ngraph::pass::GraphRewrite {
public:
    OPENVINO_RTTI("FullyConnectedBiasFusion", "0");
    FullyConnectedBiasFusion() {
        add_matcher<FullyConnectedBiasFusionWithoutMultiply>();
        add_matcher<FullyConnectedDQBiasFusion>();
    }
};

}   // namespace intel_cpu
}   // namespace ov
