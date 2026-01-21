// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Defines openvino domains for tracing
 * @file itt.hpp
 */

#pragma once

#include <openvino/itt.hpp>

namespace ov::intel_gpu {
namespace itt {
namespace domains {
    // Domain namespace to define GPU Inference phase tasks
    OV_ITT_DOMAIN(intel_gpu_inference, "ov::phases::gpu::inference");
    // Domain namespace for all of the operators
    OV_ITT_DOMAIN(intel_gpu_op, "ov::op::gpu");
    OV_ITT_DOMAIN(intel_gpu_plugin);
}  // namespace domains
}  // namespace itt
}  // namespace ov::intel_gpu
