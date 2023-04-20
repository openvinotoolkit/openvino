// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_cpu {

class NonQuantizedFullyConnectedBiasFlatten : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("NonQuantizedFullyConnectedBiasFlatten", "0");
    NonQuantizedFullyConnectedBiasFlatten();
};

class QuantizedFullyConnectedBiasFlatten : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("QuantizedFullyConnectedBiasFlatten", "0");
    QuantizedFullyConnectedBiasFlatten();
};

class FullyConnectedBiasFlatten : public ngraph::pass::GraphRewrite {
public:
    OPENVINO_RTTI("FullyConnectedBiasFlatten", "0");
    FullyConnectedBiasFlatten() {
        add_matcher<NonQuantizedFullyConnectedBiasFlatten>();
        add_matcher<QuantizedFullyConnectedBiasFlatten>();
    }
};

}   // namespace intel_cpu
}   // namespace ov
