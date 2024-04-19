// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace intel_gpu {

class ConvertFullyConnectedToFullyConnectedCompressed: public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertFullyConnectedToFullyConnectedCompressed", "0");
    ConvertFullyConnectedToFullyConnectedCompressed(bool convert_u4zp_to_u8 = false);
};

}   // namespace intel_gpu
}   // namespace ov
