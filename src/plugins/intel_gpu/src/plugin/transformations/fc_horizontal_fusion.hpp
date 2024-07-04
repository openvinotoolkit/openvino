// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace intel_gpu {

class FullyConnectedHorizontalFusion: public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("FullyConnectedHorizontalFusion", "0");
    FullyConnectedHorizontalFusion();
};

}   // namespace intel_gpu
}   // namespace ov
