// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

class ConvertLogicalXorToNotEqual : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertLogicalXorToNotEqual");
    ConvertLogicalXorToNotEqual();
};

}  // namespace ov::intel_gpu
