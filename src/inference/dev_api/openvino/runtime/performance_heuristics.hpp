// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

/**
 * @brief Performance heuristics for OpenVINO runtime
 * @file openvino/runtime/performance_heuristics.hpp
 */

#include <cfloat>
#include <memory>

#include "openvino/runtime/common.hpp"
#include "openvino/core/model.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
struct MemBandwidthPressure {
    float max_mem_tolerance = UNKNOWN;
    float ratio_compute_convs = 0;
    float ratio_mem_limited_convs = 0;
    float ratio_mem_limited_deconvs = 0;
    float ratio_mem_limited_gemms = 0;
    float ratio_compute_deconvs = 0;

    static constexpr float UNKNOWN = FLT_MAX;
    static constexpr float ALL = 1.0f;
    static constexpr float NONE = 0.0f;
    static constexpr float LIMITED = 0.5f;  // conservatively assume 1/2 utilization of the cache
};

OPENVINO_RUNTIME_API MemBandwidthPressure mem_bandwidth_pressure_tolerance(
    const std::shared_ptr<ov::Model> model,
    const float cache_size,
    const float memThresholdAssumeLimited = MemBandwidthPressure::LIMITED);

}  // namespace ov
