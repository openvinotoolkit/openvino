// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace intel_gpu {

class CheckDequantizationSubgraph : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("CheckDequantizationSubgraph", "0");
    CheckDequantizationSubgraph(const element::TypeVector& precisions);
};

}   // namespace intel_gpu
}   // namespace ov
