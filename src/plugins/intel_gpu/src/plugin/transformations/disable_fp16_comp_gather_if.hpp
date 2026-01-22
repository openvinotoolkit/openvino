// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"

namespace ov::intel_gpu {

class DisableFP16CompForDetectron2MaskRCNNGatherIfPattern: public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("DisableFP16CompForDetectron2MaskRCNNGatherIfPattern");
    DisableFP16CompForDetectron2MaskRCNNGatherIfPattern();
};

}   // namespace ov::intel_gpu
