// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "ov_ops/dynamic_quantize.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API FullyConnectedDynamicQuantization: public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FullyConnectedDynamicQuantization");
    FullyConnectedDynamicQuantization(uint64_t group_size, const ov::element::Type& quantization_dt);
};

}   // namespace pass
}   // namespace ov
