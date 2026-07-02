// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov::intel_gpu {

// Keeps the precision-sensitive NMS boundary pattern in fp32 before
// ConvertPrecision. The local box-offset construction and the NMS thresholds
// are boundary-sensitive; lowering them to fp16 can change suppression
// results.
class KeepNMSBoundaryPrecision : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::intel_gpu::KeepNMSBoundaryPrecision");
    KeepNMSBoundaryPrecision();
};

}  // namespace ov::intel_gpu