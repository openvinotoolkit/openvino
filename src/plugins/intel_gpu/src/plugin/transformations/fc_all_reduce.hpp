// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace intel_gpu {

class FullyConnectedSplitInput: public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("FullyConnectedSplitInput", "0");
    FullyConnectedSplitInput(size_t world_size, size_t rank_size);
};

}   // namespace intel_gpu
}   // namespace ov
