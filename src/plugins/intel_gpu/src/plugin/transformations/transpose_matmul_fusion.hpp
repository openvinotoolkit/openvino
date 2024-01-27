// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace intel_gpu {

class TransposeMatMulFusion: public ov::pass::GraphRewrite {
public:
    OPENVINO_RTTI("TransposeMatMulFusion", "0");
    TransposeMatMulFusion();
};

}   // namespace intel_gpu
}   // namespace ov
