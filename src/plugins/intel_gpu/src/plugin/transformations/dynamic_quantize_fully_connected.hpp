// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov::intel_gpu {

class DynamicQuantizeFullyConnected: public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("DynamicQuantizeFullyConnected");
    DynamicQuantizeFullyConnected(uint64_t group_size,
                                  bool asymmetric = false,
                                  ov::hint::DynamicQuantizationDataType dtype_scheme = ov::hint::DynamicQuantizationDataType::INT8);
};

}   // namespace ov::intel_gpu
