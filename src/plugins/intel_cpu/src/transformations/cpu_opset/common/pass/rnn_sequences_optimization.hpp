// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace intel_cpu {

class OptimizeGRUSequenceTransposes : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("OptimizeGRUSequenceTransposes", "0");
    OptimizeGRUSequenceTransposes();
};

class OptimizeLSTMSequenceTransposes : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("OptimizeLSTMSequenceTransposes", "0");
    OptimizeLSTMSequenceTransposes();
};

class OptimizeRNNSequenceTransposes : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("OptimizeRNNSequenceTransposes", "0");
    OptimizeRNNSequenceTransposes();
};

class OptimizeSequenceTransposes : public ov::pass::GraphRewrite {
public:
    OPENVINO_RTTI("OptimizeSequenceTransposes", "0");
    OptimizeSequenceTransposes();
};

}   // namespace intel_cpu
}   // namespace ov
