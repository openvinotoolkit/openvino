// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "openvino/core/visibility.hpp"

namespace ov {

/**
 * @brief Get distinct maximum-frequency group counts for physical P-, E-, and LP-E-core types.
 * @return Counts in MAIN_CORE_PROC, EFFICIENT_CORE_PROC, LP_EFFICIENT_CORE_PROC order, or empty if unavailable.
 */
std::vector<int> get_cpu_frequency_group_counts();

}  // namespace ov
