// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace intel_cpu {

class ConvertToPowerStatic: public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertToPowerStatic", "0");
    ConvertToPowerStatic();
};

}   // namespace intel_cpu
}   // namespace ov
