// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace intel_cpu {

class FullyConnectedBiasFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("FullyConnectedBiasFusion", "0");
    FullyConnectedBiasFusion();
};

class FullyConnectedBiasFusions : public ov::pass::GraphRewrite {
public:
    OPENVINO_RTTI("FullyConnectedBiasFusion", "0");
    FullyConnectedBiasFusions() {
        add_matcher<FullyConnectedBiasFusion>();
    }
};

}   // namespace intel_cpu
}   // namespace ov
