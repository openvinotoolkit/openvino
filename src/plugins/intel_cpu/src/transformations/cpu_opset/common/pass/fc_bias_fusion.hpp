// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_cpu {

class NonQuantizedFullyConnectedBiasFusion : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("NonQuantizedFullyConnectedBiasFusion", "0");
    NonQuantizedFullyConnectedBiasFusion();
};

class QuantizedFullyConnectedBiasFusion : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("FullyConnectedDQBiasFusion", "0");
    QuantizedFullyConnectedBiasFusion();
};

class FullyConnectedBiasFusion : public ngraph::pass::GraphRewrite {
public:
    OPENVINO_RTTI("FullyConnectedBiasFusion", "0");
    FullyConnectedBiasFusion() {
        add_matcher<NonQuantizedFullyConnectedBiasFusion>();
        add_matcher<QuantizedFullyConnectedBiasFusion>();
    }
};

}   // namespace intel_cpu
}   // namespace ov
