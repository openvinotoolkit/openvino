// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace intel_cpu {

class ConvertToSwishCPU: public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertToSwishCPU", "0");
    ConvertToSwishCPU();
};

}   // namespace intel_cpu
}   // namespace ov
