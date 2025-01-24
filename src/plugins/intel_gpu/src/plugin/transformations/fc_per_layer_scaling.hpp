// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

class FullyConnectedPerLayerScaling: public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FullyConnectedPerLayerScaling");
    FullyConnectedPerLayerScaling(float scale_factor);
};

}   // namespace ov::intel_gpu
