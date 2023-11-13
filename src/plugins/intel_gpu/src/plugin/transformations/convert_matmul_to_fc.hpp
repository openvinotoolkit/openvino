// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace intel_gpu {

class ConvertMatMulToFullyConnected: public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertMatMulToFullyConnected", "0");
    ConvertMatMulToFullyConnected();
};

}   // namespace intel_gpu
}   // namespace ov
