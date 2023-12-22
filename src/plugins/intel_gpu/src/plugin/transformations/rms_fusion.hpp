// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace intel_gpu {

class RMSFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("RMSFusion", "0");
    RMSFusion();
};

}   // namespace intel_gpu
}   // namespace ov
