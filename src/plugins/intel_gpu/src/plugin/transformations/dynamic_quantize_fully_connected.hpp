// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace intel_gpu {

class DynamicQuantizeFullyConnected: public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("DynamicQuantizeFullyConnected", "0");
    DynamicQuantizeFullyConnected(uint64_t group_size);
};

}   // namespace intel_gpu
}   // namespace ov
