// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace intel_cpu {

class OptimizeGRUSequenceTransposes : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("OptimizeGRUSequenceTransposes");
    OptimizeGRUSequenceTransposes();
};

class OptimizeLSTMSequenceTransposes : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("OptimizeLSTMSequenceTransposes");
    OptimizeLSTMSequenceTransposes();
};

class OptimizeRNNSequenceTransposes : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("OptimizeRNNSequenceTransposes");
    OptimizeRNNSequenceTransposes();
};

class OptimizeSequenceTransposes : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("OptimizeSequenceTransposes");
    OptimizeSequenceTransposes();
};

}  // namespace intel_cpu
}  // namespace ov
