// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"

namespace ov::intel_gpu {

class DisableFP16CompForRMS: public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("DisableFP16CompForRMS");
    DisableFP16CompForRMS();
};

}   // namespace ov::intel_gpu
