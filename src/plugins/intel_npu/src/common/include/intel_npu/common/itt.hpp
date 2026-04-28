// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/itt.hpp"

namespace intel_npu {
namespace itt {
namespace domains {

OV_ITT_DOMAIN(NPUPlugin);
OV_ITT_DOMAIN(LevelZeroBackend);
// Domain namespace to define NPU Inference phase tasks
OV_ITT_DOMAIN(InferenceNPU, "ov::phases::npu::inference");
OV_ITT_DOMAIN(NPUOps, "ov::op::npu");
}  // namespace domains
}  // namespace itt
}  // namespace intel_npu

namespace itt = ::intel_npu::itt;
