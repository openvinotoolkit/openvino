// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace intel_gpu {

class FullyConnectedConvertFusion: public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("FullyConnectedConvertFusion", "0");
    FullyConnectedConvertFusion();
};

}   // namespace intel_gpu
}   // namespace ov
