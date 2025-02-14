// Copyright (C) 2018-2025 Intel Corporation
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
    OV_ITT_DOMAIN(intel_gpu_plugin);
}  // namespace domains
}  // namespace itt
}  // namespace ov::intel_gpu
