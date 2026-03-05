// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

class ConvertExtensionOp: public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("ConvertExtensionOp");
    ConvertExtensionOp();
};

class ConvertFullyConnectedCommonToFullyConnected: public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertFullyConnectedCommonToFullyConnected");
    ConvertFullyConnectedCommonToFullyConnected();
};

}   // namespace ov::intel_gpu
