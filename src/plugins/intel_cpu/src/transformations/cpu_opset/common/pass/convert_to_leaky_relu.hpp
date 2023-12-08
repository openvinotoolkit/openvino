// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace intel_cpu {

class ConvertToLeakyRelu: public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertToLeakyRelu", "0");
    ConvertToLeakyRelu();
};

}   // namespace intel_cpu
}   // namespace ov
