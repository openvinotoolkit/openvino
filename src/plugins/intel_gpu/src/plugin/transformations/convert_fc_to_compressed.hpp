// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/device_info.hpp"
#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

class ConvertFullyConnectedToFullyConnectedCompressed: public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertFullyConnectedToFullyConnectedCompressed");
    // device_info / cm_jit_supported feed cm::woq_u2_accepts_gnk for GNK u2 weights.
    explicit ConvertFullyConnectedToFullyConnectedCompressed(const cldnn::device_info& device_info = {}, bool cm_jit_supported = false);

private:
    cldnn::device_info m_device_info;
    bool m_cm_jit_supported;
};

}   // namespace ov::intel_gpu
